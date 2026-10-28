#ifndef WOLF_MODEL_PARSER_H
#define WOLF_MODEL_PARSER_H

#include <stddef.h>
#include <stdint.h>

/* Parse a complete in-memory model file; returned strings are owned by the parser allocator.
 * 解析完整的内存模型文件；返回字符串由解析器分配器持有，调用方需用 wolf_parser_free 释放。 */
int wolf_parse_model(const uint8_t *data, size_t length, char **json_out, char **error_out);
/* Release memory returned by wolf_parse_model.
 * 释放 wolf_parse_model 返回的内存。 */
void wolf_parser_free(void *pointer);

#endif
