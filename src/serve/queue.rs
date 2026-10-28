use std::collections::BTreeSet;
use std::num::NonZeroUsize;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

#[derive(Default)]
struct QueueState {
    next_ticket: u64,
    serving: u64,
    active: usize,
    cancelled: BTreeSet<u64>,
}

pub(crate) struct FifoQueue {
    state: Mutex<QueueState>,
    changed: Condvar,
    max_active: NonZeroUsize,
}

impl FifoQueue {
    pub(crate) fn new(max_active: usize) -> Result<Self, String> {
        let max_active = NonZeroUsize::new(max_active)
            .ok_or_else(|| "Maximum concurrent generations must be greater than 0.".to_string())?;
        Ok(Self {
            state: Mutex::new(QueueState::default()),
            changed: Condvar::new(),
            max_active,
        })
    }

    pub(crate) fn enqueue(self: &Arc<Self>) -> QueueTicket {
        let mut state = self.lock_state();
        let number = state.next_ticket;
        state.next_ticket = state.next_ticket.wrapping_add(1);
        QueueTicket {
            queue: Arc::clone(self),
            number,
            acquired: false,
        }
    }

    fn acquire(self: &Arc<Self>, mut ticket: QueueTicket) -> QueueGuard {
        let mut state = self.lock_state();
        // Ticket order preserves FIFO fairness while active tracks the configured parallelism limit.
        // 票号顺序保证 FIFO 公平性，active 则记录当前并行推理数量。
        while state.serving != ticket.number || state.active >= self.max_active.get() {
            state = self
                .changed
                .wait(state)
                .unwrap_or_else(|poisoned| poisoned.into_inner());
        }
        state.active += 1;
        state.serving = state.serving.wrapping_add(1);
        advance_cancelled(&mut state);
        ticket.acquired = true;
        drop(ticket);
        self.changed.notify_all();
        QueueGuard {
            queue: Arc::clone(self),
        }
    }

    fn cancel(&self, ticket: u64) {
        let mut state = self.lock_state();
        if state.serving == ticket {
            state.serving = state.serving.wrapping_add(1);
            advance_cancelled(&mut state);
        } else {
            state.cancelled.insert(ticket);
        }
        self.changed.notify_all();
    }

    fn lock_state(&self) -> MutexGuard<'_, QueueState> {
        self.state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

fn advance_cancelled(state: &mut QueueState) {
    // Skip abandoned tickets only when they reach the head, so later requests cannot jump the queue.
    // 只有取消票到达队首时才跳过，避免后续请求越过仍在等待的请求。
    while state.cancelled.remove(&state.serving) {
        state.serving = state.serving.wrapping_add(1);
    }
}

pub(crate) struct QueueTicket {
    queue: Arc<FifoQueue>,
    number: u64,
    acquired: bool,
}

impl QueueTicket {
    pub(crate) fn acquire(self) -> QueueGuard {
        let queue = Arc::clone(&self.queue);
        queue.acquire(self)
    }
}

impl Drop for QueueTicket {
    fn drop(&mut self) {
        if !self.acquired {
            self.queue.cancel(self.number);
        }
    }
}

pub(crate) struct QueueGuard {
    queue: Arc<FifoQueue>,
}

impl Drop for QueueGuard {
    fn drop(&mut self) {
        let mut state = self.queue.lock_state();
        state.active -= 1;
        self.queue.changed.notify_all();
    }
}

impl Default for FifoQueue {
    fn default() -> Self {
        Self::new(1).expect("one is a valid generation limit")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;
    use std::time::Duration;

    #[test]
    fn serves_waiters_in_ticket_order() {
        let queue = Arc::new(FifoQueue::default());
        let first = queue.enqueue();
        let first_guard = first.acquire();
        let second = queue.enqueue();
        let third = queue.enqueue();
        let (tx, rx) = mpsc::channel();

        std::thread::scope(|scope| {
            let second_tx = tx.clone();
            scope.spawn(move || {
                let _guard = second.acquire();
                second_tx.send(2).unwrap();
            });
            scope.spawn(move || {
                let _guard = third.acquire();
                tx.send(3).unwrap();
            });
            drop(first_guard);
        });

        assert_eq!(rx.into_iter().collect::<Vec<_>>(), [2, 3]);
    }

    #[test]
    fn cancelled_request_does_not_block_following_requests() {
        let queue = Arc::new(FifoQueue::default());
        let first = queue.enqueue();
        let cancelled = queue.enqueue();
        let third = queue.enqueue();
        drop(cancelled);

        drop(first.acquire());
        drop(third.acquire());
    }

    #[test]
    fn permits_only_the_configured_number_of_active_tickets() {
        let queue = Arc::new(FifoQueue::new(2).unwrap());
        let first_guard = queue.enqueue().acquire();
        let second_guard = queue.enqueue().acquire();
        let third = queue.enqueue();
        let (started_tx, started_rx) = mpsc::channel();
        let (acquired_tx, acquired_rx) = mpsc::channel();

        std::thread::scope(|scope| {
            let worker = scope.spawn(move || {
                started_tx.send(()).unwrap();
                let _guard = third.acquire();
                acquired_tx.send(()).unwrap();
            });
            started_rx.recv().unwrap();
            assert!(acquired_rx.recv_timeout(Duration::from_millis(50)).is_err());

            drop(first_guard);
            acquired_rx
                .recv_timeout(Duration::from_secs(1))
                .expect("the third ticket should acquire the released slot");
            drop(second_guard);
            worker.join().unwrap();
        });
    }

    #[test]
    fn rejects_a_zero_generation_limit() {
        assert!(FifoQueue::new(0).is_err());
    }
}
