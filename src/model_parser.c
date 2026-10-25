#include "model_parser.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_HEADER_SIZE ((size_t)100 * 1024 * 1024)
#define MAX_ITEMS ((uint64_t)1000000)
#define MAX_JSON_DEPTH 128

typedef struct {
    char *data;
    size_t length;
    size_t capacity;
} Buffer;

typedef struct {
    char *name;
    uint64_t *shape;
    size_t dimensions;
    char *dtype;
    int has_bytes;
    uint64_t bytes;
} Tensor;

typedef struct {
    char *key;
    char *value;
} Metadata;

typedef struct {
    const char *format;
    Tensor *tensors;
    size_t tensor_count;
    size_t tensor_capacity;
    Metadata *metadata;
    size_t metadata_count;
    size_t metadata_capacity;
} Model;

typedef struct {
    uint64_t start;
    uint64_t end;
    const char *name;
    size_t order;
} Range;

typedef struct {
    const uint8_t *data;
    size_t length;
    size_t position;
    char *error;
} Reader;

typedef enum {
    JSON_NULL,
    JSON_BOOL,
    JSON_NUMBER,
    JSON_STRING,
    JSON_ARRAY,
    JSON_OBJECT
} JsonType;

typedef struct JsonValue JsonValue;

struct JsonValue {
    JsonType type;
    char *text;
    int boolean;
    JsonValue **items;
    char **keys;
    size_t count;
    size_t capacity;
};

typedef struct {
    const uint8_t *data;
    size_t length;
    size_t position;
    size_t depth;
    char *error;
} JsonReader;

static char *duplicate_string(const char *value) {
    size_t length = strlen(value);
    char *copy = (char *)malloc(length + 1);
    if (copy != NULL) {
        memcpy(copy, value, length + 1);
    }
    return copy;
}

static int internal_string_compare(const char *left, const char *right) {
    const unsigned char *a = (const unsigned char *)left;
    const unsigned char *b = (const unsigned char *)right;
    for (;;) {
        unsigned int av = a[0];
        unsigned int bv = b[0];
        size_t alen = 1;
        size_t blen = 1;
        if (a[0] == 0xff && a[1] == 0xfe) {
            av = 0;
            alen = 2;
        }
        if (b[0] == 0xff && b[1] == 0xfe) {
            bv = 0;
            blen = 2;
        }
        if (av != bv) return av < bv ? -1 : 1;
        if (av == 0) return 0;
        a += alen;
        b += blen;
    }
}

static char *duplicate_bytes(const uint8_t *value, size_t length) {
    size_t index;
    size_t extra = 0;
    size_t output_index = 0;
    char *copy;
    for (index = 0; index < length; index++) {
        if (value[index] == 0) extra++;
    }
    if (length > SIZE_MAX - extra - 1) return NULL;
    copy = (char *)malloc(length + extra + 1);
    if (copy == NULL) return NULL;
    for (index = 0; index < length; index++) {
        if (value[index] == 0) {
            copy[output_index++] = (char)0xff;
            copy[output_index++] = (char)0xfe;
        } else {
            copy[output_index++] = (char)value[index];
        }
    }
    copy[output_index] = '\0';
    return copy;
}

static void set_error(char **target, const char *format, ...) {
    va_list args;
    va_list copy;
    int length;
    char *message;

    if (*target != NULL) {
        return;
    }
    va_start(args, format);
    va_copy(copy, args);
    length = vsnprintf(NULL, 0, format, copy);
    va_end(copy);
    if (length < 0) {
        va_end(args);
        *target = duplicate_string("Parsing failed.");
        return;
    }
    message = (char *)malloc((size_t)length + 1);
    if (message != NULL) {
        (void)vsnprintf(message, (size_t)length + 1, format, args);
    }
    va_end(args);
    *target = message != NULL ? message : duplicate_string("Memory allocation failed.");
}

static int buffer_reserve(Buffer *buffer, size_t extra) {
    size_t needed;
    size_t capacity;
    char *data;

    if (extra > SIZE_MAX - buffer->length - 1) {
        return 0;
    }
    needed = buffer->length + extra + 1;
    if (needed <= buffer->capacity) {
        return 1;
    }
    capacity = buffer->capacity == 0 ? 128 : buffer->capacity;
    while (capacity < needed) {
        if (capacity > SIZE_MAX / 2) {
            capacity = needed;
            break;
        }
        capacity *= 2;
    }
    data = (char *)realloc(buffer->data, capacity);
    if (data == NULL) {
        return 0;
    }
    buffer->data = data;
    buffer->capacity = capacity;
    return 1;
}

static int buffer_append_n(Buffer *buffer, const char *value, size_t length) {
    if (!buffer_reserve(buffer, length)) {
        return 0;
    }
    memcpy(buffer->data + buffer->length, value, length);
    buffer->length += length;
    buffer->data[buffer->length] = '\0';
    return 1;
}

static int buffer_append(Buffer *buffer, const char *value) {
    return buffer_append_n(buffer, value, strlen(value));
}

static int buffer_append_char(Buffer *buffer, char value) {
    return buffer_append_n(buffer, &value, 1);
}

static int buffer_append_format(Buffer *buffer, const char *format, ...) {
    va_list args;
    va_list copy;
    int length;
    int result;

    va_start(args, format);
    va_copy(copy, args);
    length = vsnprintf(NULL, 0, format, copy);
    va_end(copy);
    if (length < 0 || !buffer_reserve(buffer, (size_t)length)) {
        va_end(args);
        return 0;
    }
    result = vsnprintf(buffer->data + buffer->length,
                       buffer->capacity - buffer->length, format, args);
    va_end(args);
    if (result < 0) {
        return 0;
    }
    buffer->length += (size_t)result;
    return 1;
}

static int buffer_append_json_string(Buffer *buffer, const char *value) {
    const unsigned char *cursor = (const unsigned char *)value;
    if (!buffer_append_char(buffer, '"')) {
        return 0;
    }
    while (*cursor != 0) {
        if (cursor[0] == 0xff && cursor[1] == 0xfe) {
            if (!buffer_append(buffer, "\\u0000")) return 0;
            cursor += 2;
            continue;
        }
        unsigned char ch = *cursor++;
        switch (ch) {
        case '"':
            if (!buffer_append(buffer, "\\\"")) return 0;
            break;
        case '\\':
            if (!buffer_append(buffer, "\\\\")) return 0;
            break;
        case '\b':
            if (!buffer_append(buffer, "\\b")) return 0;
            break;
        case '\f':
            if (!buffer_append(buffer, "\\f")) return 0;
            break;
        case '\n':
            if (!buffer_append(buffer, "\\n")) return 0;
            break;
        case '\r':
            if (!buffer_append(buffer, "\\r")) return 0;
            break;
        case '\t':
            if (!buffer_append(buffer, "\\t")) return 0;
            break;
        default:
            if (ch < 0x20) {
                if (!buffer_append_format(buffer, "\\u%04x", ch)) return 0;
            } else if (!buffer_append_char(buffer, (char)ch)) {
                return 0;
            }
        }
    }
    return buffer_append_char(buffer, '"');
}

static void free_model(Model *model) {
    size_t index;
    for (index = 0; index < model->tensor_count; index++) {
        free(model->tensors[index].name);
        free(model->tensors[index].shape);
        free(model->tensors[index].dtype);
    }
    for (index = 0; index < model->metadata_count; index++) {
        free(model->metadata[index].key);
        free(model->metadata[index].value);
    }
    free(model->tensors);
    free(model->metadata);
    memset(model, 0, sizeof(*model));
}

static int grow_array(void **items, size_t *capacity, size_t count, size_t item_size) {
    size_t new_capacity;
    void *new_items;
    if (count < *capacity) {
        return 1;
    }
    new_capacity = *capacity == 0 ? 16 : *capacity * 2;
    if (new_capacity < *capacity || new_capacity > SIZE_MAX / item_size) {
        return 0;
    }
    new_items = realloc(*items, new_capacity * item_size);
    if (new_items == NULL) {
        return 0;
    }
    *items = new_items;
    *capacity = new_capacity;
    return 1;
}

static int model_add_tensor(Model *model, Tensor *tensor) {
    if (!grow_array((void **)&model->tensors, &model->tensor_capacity,
                    model->tensor_count, sizeof(Tensor))) {
        return 0;
    }
    model->tensors[model->tensor_count++] = *tensor;
    memset(tensor, 0, sizeof(*tensor));
    return 1;
}

static int model_add_metadata(Model *model, const char *key, const char *value) {
    Metadata metadata;
    metadata.key = duplicate_string(key);
    metadata.value = duplicate_string(value);
    if (metadata.key == NULL || metadata.value == NULL ||
        !grow_array((void **)&model->metadata, &model->metadata_capacity,
                    model->metadata_count, sizeof(Metadata))) {
        free(metadata.key);
        free(metadata.value);
        return 0;
    }
    model->metadata[model->metadata_count++] = metadata;
    return 1;
}

static int valid_utf8(const uint8_t *data, size_t length) {
    size_t index = 0;
    while (index < length) {
        uint8_t first = data[index++];
        uint32_t codepoint;
        size_t continuation;
        if (first <= 0x7f) continue;
        if (first >= 0xc2 && first <= 0xdf) {
            codepoint = first & 0x1f;
            continuation = 1;
        } else if (first >= 0xe0 && first <= 0xef) {
            codepoint = first & 0x0f;
            continuation = 2;
        } else if (first >= 0xf0 && first <= 0xf4) {
            codepoint = first & 0x07;
            continuation = 3;
        } else {
            return 0;
        }
        if (continuation > length - index) return 0;
        while (continuation-- > 0) {
            uint8_t next = data[index++];
            if ((next & 0xc0) != 0x80) return 0;
            codepoint = (codepoint << 6) | (next & 0x3f);
        }
        if (codepoint < 0x80 || (codepoint < 0x800 && first >= 0xe0) ||
            (codepoint < 0x10000 && first >= 0xf0) ||
            codepoint > 0x10ffff || (codepoint >= 0xd800 && codepoint <= 0xdfff)) {
            return 0;
        }
    }
    return 1;
}

static int reader_take(Reader *reader, size_t count, const uint8_t **value) {
    if (count > SIZE_MAX - reader->position ||
        reader->position + count > reader->length) {
        set_error(&reader->error, "Unexpected end of GGUF file; the file may be corrupted.");
        return 0;
    }
    *value = reader->data + reader->position;
    reader->position += count;
    return 1;
}

static int reader_u8(Reader *reader, uint8_t *value) {
    const uint8_t *bytes;
    if (!reader_take(reader, 1, &bytes)) return 0;
    *value = bytes[0];
    return 1;
}

static int reader_u16(Reader *reader, uint16_t *value) {
    const uint8_t *bytes;
    if (!reader_take(reader, 2, &bytes)) return 0;
    *value = (uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8);
    return 1;
}

static int reader_u32(Reader *reader, uint32_t *value) {
    const uint8_t *bytes;
    if (!reader_take(reader, 4, &bytes)) return 0;
    *value = (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8) |
             ((uint32_t)bytes[2] << 16) | ((uint32_t)bytes[3] << 24);
    return 1;
}

static int reader_u64(Reader *reader, uint64_t *value) {
    const uint8_t *bytes;
    size_t index;
    *value = 0;
    if (!reader_take(reader, 8, &bytes)) return 0;
    for (index = 0; index < 8; index++) {
        *value |= (uint64_t)bytes[index] << (index * 8);
    }
    return 1;
}

static int reader_count(Reader *reader, const char *label, uint64_t *count) {
    if (!reader_u64(reader, count)) return 0;
    if (*count > MAX_ITEMS) {
        set_error(&reader->error, "Invalid number of GGUF %s: %" PRIu64 ".", label, *count);
        return 0;
    }
    return 1;
}

static int reader_string(Reader *reader, char **value) {
    uint64_t length;
    const uint8_t *bytes;
    char *copy;
    if (!reader_u64(reader, &length)) return 0;
    if (length > MAX_HEADER_SIZE) {
        set_error(&reader->error, "GGUF string length exceeds the safety limit.");
        return 0;
    }
    if (length > SIZE_MAX - 1 || !reader_take(reader, (size_t)length, &bytes)) {
        if (reader->error == NULL) {
            set_error(&reader->error, "GGUF string length exceeds the range supported by this platform.");
        }
        return 0;
    }
    if (!valid_utf8(bytes, (size_t)length)) {
        set_error(&reader->error, "GGUF file contains an invalid UTF-8 string.");
        return 0;
    }
    copy = duplicate_bytes(bytes, (size_t)length);
    if (copy == NULL) {
        set_error(&reader->error, "Memory allocation failed.");
        return 0;
    }
    *value = copy;
    return 1;
}

static int format_integer(char **value, const char *format, ...) {
    va_list args;
    va_list copy;
    int length;
    va_start(args, format);
    va_copy(copy, args);
    length = vsnprintf(NULL, 0, format, copy);
    va_end(copy);
    if (length >= 0) {
        *value = (char *)malloc((size_t)length + 1);
        if (*value != NULL) {
            (void)vsnprintf(*value, (size_t)length + 1, format, args);
        }
    }
    va_end(args);
    return length >= 0 && *value != NULL;
}

static int reader_metadata_value(Reader *reader, uint32_t type, char **value,
                                 unsigned int depth);

static int reader_metadata_value(Reader *reader, uint32_t type, char **value,
                                 unsigned int depth) {
    uint8_t u8;
    uint16_t u16;
    uint32_t u32;
    uint64_t u64;
    if (depth > MAX_JSON_DEPTH) {
        set_error(&reader->error, "GGUF metadata array is nested too deeply.");
        return 0;
    }
    switch (type) {
    case 0:
        if (!reader_u8(reader, &u8)) return 0;
        return format_integer(value, "%u", (unsigned int)u8);
    case 1:
        if (!reader_u8(reader, &u8)) return 0;
        return format_integer(value, "%d", (int)(int8_t)u8);
    case 2:
        if (!reader_u16(reader, &u16)) return 0;
        return format_integer(value, "%u", (unsigned int)u16);
    case 3:
        if (!reader_u16(reader, &u16)) return 0;
        return format_integer(value, "%d", (int)(int16_t)u16);
    case 4:
        if (!reader_u32(reader, &u32)) return 0;
        return format_integer(value, "%" PRIu32, u32);
    case 5:
        if (!reader_u32(reader, &u32)) return 0;
        return format_integer(value, "%" PRId32, (int32_t)u32);
    case 6: {
        uint32_t bits;
        float number;
        if (!reader_u32(reader, &bits)) return 0;
        memcpy(&number, &bits, sizeof(number));
        return format_integer(value, "%.9g", (double)number);
    }
    case 7:
        if (!reader_u8(reader, &u8)) return 0;
        *value = duplicate_string(u8 != 0 ? "true" : "false");
        return *value != NULL;
    case 8:
        return reader_string(reader, value);
    case 9: {
        uint32_t element_type;
        uint64_t count;
        uint64_t index;
        char count_text[64];
        if (!reader_u32(reader, &element_type) ||
            !reader_count(reader, "array elements", &count)) return 0;
        for (index = 0; index < count; index++) {
            char *ignored = NULL;
            if (!reader_metadata_value(reader, element_type, &ignored, depth + 1)) {
                free(ignored);
                return 0;
            }
            free(ignored);
        }
        (void)snprintf(count_text, sizeof(count_text), "[%" PRIu64 " items]", count);
        *value = duplicate_string(count_text);
        return *value != NULL;
    }
    case 10:
        if (!reader_u64(reader, &u64)) return 0;
        return format_integer(value, "%" PRIu64, u64);
    case 11:
        if (!reader_u64(reader, &u64)) return 0;
        return format_integer(value, "%" PRId64, (int64_t)u64);
    case 12: {
        uint64_t bits;
        double number;
        if (!reader_u64(reader, &bits)) return 0;
        memcpy(&number, &bits, sizeof(number));
        return format_integer(value, "%.17g", number);
    }
    default:
        set_error(&reader->error, "GGUF metadata contains unknown type %" PRIu32 ".", type);
        return 0;
    }
}

static const char *ggml_type_name(uint32_t type) {
    switch (type) {
    case 0: return "F32";
    case 1: return "F16";
    case 2: return "Q4_0";
    case 3: return "Q4_1";
    case 6: return "Q5_0";
    case 7: return "Q5_1";
    case 8: return "Q8_0";
    case 9: return "Q8_1";
    case 10: return "Q2_K";
    case 11: return "Q3_K";
    case 12: return "Q4_K";
    case 13: return "Q5_K";
    case 14: return "Q6_K";
    case 15: return "Q8_K";
    case 16: return "IQ2_XXS";
    case 17: return "IQ2_XS";
    case 18: return "IQ3_XXS";
    case 19: return "IQ1_S";
    case 20: return "IQ4_NL";
    case 21: return "IQ3_S";
    case 22: return "IQ2_S";
    case 23: return "IQ4_XS";
    case 24: return "I8";
    case 25: return "I16";
    case 26: return "I32";
    case 27: return "I64";
    case 28: return "F64";
    case 29: return "IQ1_M";
    case 30: return "BF16";
    case 31: return "Q4_0_4_4";
    case 32: return "Q4_0_4_8";
    case 33: return "Q4_0_8_8";
    default: return "UNKNOWN";
    }
}

static int ggml_type_size(uint32_t type, uint64_t *block_size, uint64_t *type_size) {
    switch (type) {
    case 0: *block_size = 1; *type_size = 4; return 1;
    case 1: *block_size = 1; *type_size = 2; return 1;
    case 2: *block_size = 32; *type_size = 18; return 1;
    case 3: *block_size = 32; *type_size = 20; return 1;
    case 6: *block_size = 32; *type_size = 22; return 1;
    case 7: *block_size = 32; *type_size = 24; return 1;
    case 8: *block_size = 32; *type_size = 34; return 1;
    case 9: *block_size = 32; *type_size = 36; return 1;
    case 10: *block_size = 256; *type_size = 84; return 1;
    case 11: *block_size = 256; *type_size = 110; return 1;
    case 12: *block_size = 256; *type_size = 144; return 1;
    case 13: *block_size = 256; *type_size = 176; return 1;
    case 14: *block_size = 256; *type_size = 210; return 1;
    case 16: *block_size = 256; *type_size = 66; return 1;
    case 17: *block_size = 256; *type_size = 74; return 1;
    case 18: *block_size = 256; *type_size = 98; return 1;
    case 19: *block_size = 256; *type_size = 50; return 1;
    case 20: *block_size = 32; *type_size = 20; return 1;
    case 21: *block_size = 256; *type_size = 110; return 1;
    case 22: *block_size = 256; *type_size = 65; return 1;
    case 23: *block_size = 256; *type_size = 72; return 1;
    case 24: *block_size = 1; *type_size = 1; return 1;
    case 25: *block_size = 1; *type_size = 2; return 1;
    case 26: *block_size = 1; *type_size = 4; return 1;
    case 27: *block_size = 1; *type_size = 8; return 1;
    case 28: *block_size = 1; *type_size = 8; return 1;
    case 29: *block_size = 256; *type_size = 56; return 1;
    case 30: *block_size = 1; *type_size = 2; return 1;
    case 31:
    case 32:
    case 33: *block_size = 32; *type_size = 18; return 1;
    default: return 0;
    }
}

static int align_up(uint64_t value, uint64_t alignment, uint64_t *result) {
    if (value > UINT64_MAX - (alignment - 1)) return 0;
    *result = (value + alignment - 1) & ~(alignment - 1);
    return 1;
}

static int compare_ranges(const void *left, const void *right) {
    const Range *a = (const Range *)left;
    const Range *b = (const Range *)right;
    if (a->start < b->start) return -1;
    if (a->start > b->start) return 1;
    if (a->order < b->order) return -1;
    if (a->order > b->order) return 1;
    return 0;
}

static int parse_gguf(const uint8_t *data, size_t length, Model *model, char **error) {
    Reader reader = {data, length, 0, NULL};
    uint32_t version;
    uint64_t tensor_count;
    uint64_t metadata_count;
    uint64_t alignment = 32;
    uint64_t index;
    Range *ranges = NULL;
    size_t range_count = 0;
    size_t range_capacity = 0;
    uint64_t data_start;
    uint64_t data_length;
    const uint8_t *signature;

    model->format = "GGUF";
    if (!reader_take(&reader, 4, &signature)) goto failed;
    if (memcmp(signature, "GGUF", 4) != 0) {
        set_error(&reader.error, "Invalid GGUF file signature.");
        goto failed;
    }
    if (!reader_u32(&reader, &version)) goto failed;
    if (version < 2 || version > 3) {
        set_error(&reader.error, "Unsupported GGUF version %" PRIu32 "; supported versions are 2 and 3.", version);
        goto failed;
    }
    if (!reader_count(&reader, "tensors", &tensor_count) ||
        !reader_count(&reader, "metadata entries", &metadata_count)) goto failed;

    for (index = 0; index < metadata_count; index++) {
        char *key = NULL;
        char *value = NULL;
        uint32_t type;
        if (!reader_string(&reader, &key) || !reader_u32(&reader, &type) ||
            !reader_metadata_value(&reader, type, &value, 0)) {
            free(key);
            free(value);
            goto failed;
        }
        if (internal_string_compare(key, "general.alignment") == 0) {
            char *end = NULL;
            unsigned long long parsed;
            errno = 0;
            parsed = strtoull(value, &end, 10);
            if (errno != 0 || end == value || *end != '\0' || value[0] == '-' ||
                value[0] == ' ' || value[0] == '\t' || value[0] == '\r' ||
                value[0] == '\n' || parsed == 0 ||
                (parsed & (parsed - 1)) != 0) {
                set_error(&reader.error, "Invalid GGUF general.alignment value.");
                free(key);
                free(value);
                goto failed;
            }
            alignment = (uint64_t)parsed;
        }
        if (!model_add_metadata(model, key, value)) {
            set_error(&reader.error, "Memory allocation failed.");
            free(key);
            free(value);
            goto failed;
        }
        free(key);
        free(value);
    }

    for (index = 0; index < tensor_count; index++) {
        Tensor tensor = {0};
        uint32_t dimensions;
        uint32_t type;
        uint64_t offset;
        uint64_t block_size;
        uint64_t type_size;
        uint64_t elements = 1;
        uint64_t bytes;
        size_t dimension;
        if (!reader_string(&reader, &tensor.name) ||
            !reader_u32(&reader, &dimensions)) {
            free(tensor.name);
            goto failed;
        }
        if (dimensions > 4) {
            set_error(&reader.error, "GGUF tensor '%s' has %" PRIu32 " dimensions, exceeding the supported limit.",
                      tensor.name, dimensions);
            free(tensor.name);
            goto failed;
        }
        tensor.dimensions = dimensions;
        if (dimensions != 0) {
            tensor.shape = (uint64_t *)calloc(dimensions, sizeof(uint64_t));
            if (tensor.shape == NULL) {
                set_error(&reader.error, "Memory allocation failed.");
                free(tensor.name);
                goto failed;
            }
        }
        for (dimension = 0; dimension < dimensions; dimension++) {
            if (!reader_u64(&reader, &tensor.shape[dimension])) {
                free(tensor.name);
                free(tensor.shape);
                goto failed;
            }
        }
        if (!reader_u32(&reader, &type) || !reader_u64(&reader, &offset)) {
            free(tensor.name);
            free(tensor.shape);
            goto failed;
        }
        tensor.dtype = duplicate_string(ggml_type_name(type));
        if (tensor.dtype == NULL) {
            set_error(&reader.error, "Memory allocation failed.");
            free(tensor.name);
            free(tensor.shape);
            goto failed;
        }
        if (ggml_type_size(type, &block_size, &type_size)) {
            for (dimension = 0; dimension < dimensions; dimension++) {
                if (tensor.shape[dimension] != 0 &&
                    elements > UINT64_MAX / tensor.shape[dimension]) {
                    set_error(&reader.error, "Size overflow for GGUF tensor '%s'.", tensor.name);
                    free(tensor.name);
                    free(tensor.shape);
                    free(tensor.dtype);
                    goto failed;
                }
                elements *= tensor.shape[dimension];
            }
            if (elements == 0) elements = 1;
            if (elements > UINT64_MAX - (block_size - 1)) {
                set_error(&reader.error, "Size overflow for GGUF tensor '%s'.", tensor.name);
                free(tensor.name);
                free(tensor.shape);
                free(tensor.dtype);
                goto failed;
            }
            elements = (elements + block_size - 1) / block_size;
            if (elements > UINT64_MAX / type_size) {
                set_error(&reader.error, "Size overflow for GGUF tensor '%s'.", tensor.name);
                free(tensor.name);
                free(tensor.shape);
                free(tensor.dtype);
                goto failed;
            }
            bytes = elements * type_size;
            tensor.has_bytes = 1;
            tensor.bytes = bytes;
            if (!grow_array((void **)&ranges, &range_capacity, range_count, sizeof(Range))) {
                set_error(&reader.error, "Memory allocation failed.");
                free(tensor.name);
                free(tensor.shape);
                free(tensor.dtype);
                goto failed;
            }
            ranges[range_count] = (Range){offset, offset + bytes, tensor.name, range_count};
            range_count++;
            if (offset > UINT64_MAX - bytes) {
                set_error(&reader.error, "Size overflow for GGUF tensor '%s'.", tensor.name);
                free(tensor.name);
                free(tensor.shape);
                free(tensor.dtype);
                goto failed;
            }
            ranges[range_count - 1].end = offset + bytes;
        }
        if (!model_add_tensor(model, &tensor)) {
            set_error(&reader.error, "Memory allocation failed.");
            free(tensor.name);
            free(tensor.shape);
            free(tensor.dtype);
            goto failed;
        }
    }

    if (!align_up((uint64_t)reader.position, alignment, &data_start)) {
        set_error(&reader.error, "GGUF data section offset overflow.");
        goto failed;
    }
    if (data_start > length) {
        set_error(&reader.error, "GGUF tensor data section exceeds the file bounds.");
        goto failed;
    }
    data_length = (uint64_t)length - data_start;
    if (range_count > 1) qsort(ranges, range_count, sizeof(Range), compare_ranges);
    for (index = 0; index < range_count; index++) {
        if (ranges[index].start > data_length) {
            set_error(&reader.error, "Data offset for GGUF tensor '%s' exceeds the file bounds.", ranges[index].name);
            goto failed;
        }
        if (ranges[index].end > data_length) {
            set_error(&reader.error, "Data for GGUF tensor '%s' exceeds the file bounds.", ranges[index].name);
            goto failed;
        }
        if (index > 0 && ranges[index - 1].end > ranges[index].start) {
            set_error(&reader.error, "Data ranges for GGUF tensors '%s' and '%s' overlap.",
                      ranges[index - 1].name, ranges[index].name);
            goto failed;
        }
    }
    free(ranges);
    free(reader.error);
    return 1;

failed:
    free(ranges);
    *error = reader.error != NULL ? reader.error : duplicate_string("Memory allocation failed.");
    return 0;
}

static void json_free(JsonValue *value) {
    size_t index;
    if (value == NULL) return;
    free(value->text);
    for (index = 0; index < value->count; index++) {
        json_free(value->items[index]);
        if (value->keys != NULL) free(value->keys[index]);
    }
    free(value->items);
    free(value->keys);
    free(value);
}

static JsonValue *json_new(JsonType type) {
    JsonValue *value = (JsonValue *)calloc(1, sizeof(JsonValue));
    if (value != NULL) value->type = type;
    return value;
}

static int json_array_add(JsonValue *array, JsonValue *item) {
    if (!grow_array((void **)&array->items, &array->capacity,
                    array->count, sizeof(JsonValue *))) return 0;
    array->items[array->count++] = item;
    return 1;
}

static int json_object_set(JsonValue *object, char *key, JsonValue *item) {
    size_t index;
    for (index = 0; index < object->count; index++) {
        if (internal_string_compare(object->keys[index], key) == 0) {
            free(key);
            json_free(object->items[index]);
            object->items[index] = item;
            return 1;
        }
    }
    if (!grow_array((void **)&object->items, &object->capacity,
                    object->count, sizeof(JsonValue *))) return 0;
    {
        char **keys = (char **)realloc(object->keys, object->capacity * sizeof(char *));
        if (keys == NULL) return 0;
        object->keys = keys;
    }
    object->keys[object->count] = key;
    object->items[object->count++] = item;
    return 1;
}

static void json_skip_whitespace(JsonReader *reader) {
    while (reader->position < reader->length) {
        uint8_t ch = reader->data[reader->position];
        if (ch != ' ' && ch != '\t' && ch != '\r' && ch != '\n') break;
        reader->position++;
    }
}

static int buffer_append_codepoint(Buffer *buffer, uint32_t codepoint) {
    if (codepoint <= 0x7f) return buffer_append_char(buffer, (char)codepoint);
    if (codepoint <= 0x7ff) {
        return buffer_append_char(buffer, (char)(0xc0 | (codepoint >> 6))) &&
               buffer_append_char(buffer, (char)(0x80 | (codepoint & 0x3f)));
    }
    if (codepoint <= 0xffff) {
        return buffer_append_char(buffer, (char)(0xe0 | (codepoint >> 12))) &&
               buffer_append_char(buffer, (char)(0x80 | ((codepoint >> 6) & 0x3f))) &&
               buffer_append_char(buffer, (char)(0x80 | (codepoint & 0x3f)));
    }
    return buffer_append_char(buffer, (char)(0xf0 | (codepoint >> 18))) &&
           buffer_append_char(buffer, (char)(0x80 | ((codepoint >> 12) & 0x3f))) &&
           buffer_append_char(buffer, (char)(0x80 | ((codepoint >> 6) & 0x3f))) &&
           buffer_append_char(buffer, (char)(0x80 | (codepoint & 0x3f)));
}

static int json_hex4(JsonReader *reader, uint32_t *value) {
    size_t index;
    *value = 0;
    for (index = 0; index < 4; index++) {
        uint8_t ch;
        unsigned int digit;
        if (reader->position >= reader->length) return 0;
        ch = reader->data[reader->position++];
        if (ch >= '0' && ch <= '9') digit = ch - '0';
        else if (ch >= 'a' && ch <= 'f') digit = ch - 'a' + 10;
        else if (ch >= 'A' && ch <= 'F') digit = ch - 'A' + 10;
        else return 0;
        *value = (*value << 4) | digit;
    }
    return 1;
}

static char *json_parse_string(JsonReader *reader) {
    Buffer buffer = {0};
    if (reader->position >= reader->length || reader->data[reader->position++] != '"') {
        set_error(&reader->error, "Invalid Safetensors header JSON.");
        return NULL;
    }
    while (reader->position < reader->length) {
        uint8_t ch = reader->data[reader->position++];
        if (ch == '"') {
            if (buffer.data == NULL) {
                buffer.data = duplicate_string("");
            }
            if (buffer.data == NULL) set_error(&reader->error, "Memory allocation failed.");
            return buffer.data;
        }
        if (ch < 0x20) {
            set_error(&reader->error, "Invalid Safetensors header JSON: string contains a control character.");
            free(buffer.data);
            return NULL;
        }
        if (ch == '\\') {
            uint32_t codepoint;
            if (reader->position >= reader->length) break;
            ch = reader->data[reader->position++];
            switch (ch) {
            case '"': case '\\': case '/':
                if (!buffer_append_char(&buffer, (char)ch)) goto allocation_failure;
                break;
            case 'b': if (!buffer_append_char(&buffer, '\b')) goto allocation_failure; break;
            case 'f': if (!buffer_append_char(&buffer, '\f')) goto allocation_failure; break;
            case 'n': if (!buffer_append_char(&buffer, '\n')) goto allocation_failure; break;
            case 'r': if (!buffer_append_char(&buffer, '\r')) goto allocation_failure; break;
            case 't': if (!buffer_append_char(&buffer, '\t')) goto allocation_failure; break;
            case 'u':
                if (!json_hex4(reader, &codepoint)) goto invalid_string;
                if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
                    uint32_t low;
                    if (reader->length - reader->position < 6 ||
                        reader->data[reader->position] != '\\' ||
                        reader->data[reader->position + 1] != 'u') goto invalid_string;
                    reader->position += 2;
                    if (!json_hex4(reader, &low) || low < 0xdc00 || low > 0xdfff) {
                        goto invalid_string;
                    }
                    codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + (low - 0xdc00);
                } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) {
                    goto invalid_string;
                }
                if (codepoint == 0) {
                    if (!buffer_append_char(&buffer, (char)0xff) ||
                        !buffer_append_char(&buffer, (char)0xfe)) {
                        goto allocation_failure;
                    }
                } else if (!buffer_append_codepoint(&buffer, codepoint)) {
                    goto allocation_failure;
                }
                break;
            default:
                goto invalid_string;
            }
        } else if (ch < 0x80) {
            if (!buffer_append_char(&buffer, (char)ch)) goto allocation_failure;
        } else {
            size_t start = reader->position - 1;
            size_t count;
            if ((ch & 0xe0) == 0xc0) count = 2;
            else if ((ch & 0xf0) == 0xe0) count = 3;
            else if ((ch & 0xf8) == 0xf0) count = 4;
            else goto invalid_string;
            if (count - 1 > reader->length - reader->position ||
                !valid_utf8(reader->data + start, count)) goto invalid_string;
            reader->position = start + count;
            if (!buffer_append_n(&buffer, (const char *)reader->data + start, count)) {
                goto allocation_failure;
            }
        }
    }
invalid_string:
    set_error(&reader->error, "Invalid Safetensors header JSON: unterminated string or invalid escape.");
    free(buffer.data);
    return NULL;
allocation_failure:
    set_error(&reader->error, "Memory allocation failed.");
    free(buffer.data);
    return NULL;
}

static JsonValue *json_parse_value(JsonReader *reader);

static JsonValue *json_parse_array(JsonReader *reader) {
    JsonValue *array = json_new(JSON_ARRAY);
    reader->position++;
    json_skip_whitespace(reader);
    if (array == NULL) goto allocation_failure;
    if (reader->position < reader->length && reader->data[reader->position] == ']') {
        reader->position++;
        return array;
    }
    for (;;) {
        JsonValue *item = json_parse_value(reader);
        if (item == NULL || !json_array_add(array, item)) {
            json_free(item);
            json_free(array);
            if (reader->error == NULL) set_error(&reader->error, "Memory allocation failed.");
            return NULL;
        }
        json_skip_whitespace(reader);
        if (reader->position >= reader->length) break;
        if (reader->data[reader->position] == ']') {
            reader->position++;
            return array;
        }
        if (reader->data[reader->position++] != ',') break;
        json_skip_whitespace(reader);
    }
    json_free(array);
    set_error(&reader->error, "Invalid Safetensors header JSON: malformed array.");
    return NULL;
allocation_failure:
    set_error(&reader->error, "Memory allocation failed.");
    return NULL;
}

static JsonValue *json_parse_object(JsonReader *reader) {
    JsonValue *object = json_new(JSON_OBJECT);
    reader->position++;
    json_skip_whitespace(reader);
    if (object == NULL) goto allocation_failure;
    if (reader->position < reader->length && reader->data[reader->position] == '}') {
        reader->position++;
        return object;
    }
    for (;;) {
        char *key;
        JsonValue *item;
        if (reader->position >= reader->length || reader->data[reader->position] != '"') {
            break;
        }
        key = json_parse_string(reader);
        if (key == NULL) break;
        json_skip_whitespace(reader);
        if (reader->position >= reader->length || reader->data[reader->position++] != ':') {
            free(key);
            break;
        }
        json_skip_whitespace(reader);
        item = json_parse_value(reader);
        if (item == NULL || !json_object_set(object, key, item)) {
            free(key);
            json_free(item);
            json_free(object);
            if (reader->error == NULL) set_error(&reader->error, "Memory allocation failed.");
            return NULL;
        }
        json_skip_whitespace(reader);
        if (reader->position >= reader->length) break;
        if (reader->data[reader->position] == '}') {
            reader->position++;
            return object;
        }
        if (reader->data[reader->position++] != ',') break;
        json_skip_whitespace(reader);
    }
    json_free(object);
    if (reader->error == NULL) {
        set_error(&reader->error, "Invalid Safetensors header JSON: malformed object.");
    }
    return NULL;
allocation_failure:
    set_error(&reader->error, "Memory allocation failed.");
    return NULL;
}

static JsonValue *json_parse_number(JsonReader *reader) {
    size_t start = reader->position;
    JsonValue *number;
    if (reader->data[reader->position] == '-') reader->position++;
    if (reader->position >= reader->length) goto invalid_number;
    if (reader->data[reader->position] == '0') {
        reader->position++;
        if (reader->position < reader->length &&
            reader->data[reader->position] >= '0' && reader->data[reader->position] <= '9') {
            goto invalid_number;
        }
    } else if (reader->data[reader->position] >= '1' &&
               reader->data[reader->position] <= '9') {
        while (reader->position < reader->length &&
               reader->data[reader->position] >= '0' && reader->data[reader->position] <= '9') {
            reader->position++;
        }
    } else {
        goto invalid_number;
    }
    if (reader->position < reader->length && reader->data[reader->position] == '.') {
        reader->position++;
        if (reader->position >= reader->length ||
            reader->data[reader->position] < '0' || reader->data[reader->position] > '9') {
            goto invalid_number;
        }
        while (reader->position < reader->length &&
               reader->data[reader->position] >= '0' && reader->data[reader->position] <= '9') {
            reader->position++;
        }
    }
    if (reader->position < reader->length &&
        (reader->data[reader->position] == 'e' || reader->data[reader->position] == 'E')) {
        reader->position++;
        if (reader->position < reader->length &&
            (reader->data[reader->position] == '+' || reader->data[reader->position] == '-')) {
            reader->position++;
        }
        if (reader->position >= reader->length ||
            reader->data[reader->position] < '0' || reader->data[reader->position] > '9') {
            goto invalid_number;
        }
        while (reader->position < reader->length &&
               reader->data[reader->position] >= '0' && reader->data[reader->position] <= '9') {
            reader->position++;
        }
    }
    number = json_new(JSON_NUMBER);
    if (number == NULL) goto allocation_failure;
    number->text = (char *)malloc(reader->position - start + 1);
    if (number->text == NULL) {
        json_free(number);
        goto allocation_failure;
    }
    memcpy(number->text, reader->data + start, reader->position - start);
    number->text[reader->position - start] = '\0';
    return number;
invalid_number:
    set_error(&reader->error, "Invalid Safetensors header JSON: malformed number.");
    return NULL;
allocation_failure:
    set_error(&reader->error, "Memory allocation failed.");
    return NULL;
}

static JsonValue *json_parse_value(JsonReader *reader) {
    JsonValue *value;
    if (++reader->depth > MAX_JSON_DEPTH) {
        reader->depth--;
        set_error(&reader->error, "Safetensors header JSON is nested too deeply.");
        return NULL;
    }
    json_skip_whitespace(reader);
    if (reader->position >= reader->length) {
        set_error(&reader->error, "Invalid Safetensors header JSON: incomplete input.");
        reader->depth--;
        return NULL;
    }
    switch (reader->data[reader->position]) {
    case '{': value = json_parse_object(reader); break;
    case '[': value = json_parse_array(reader); break;
    case '"':
        value = json_new(JSON_STRING);
        if (value != NULL) {
            value->text = json_parse_string(reader);
            if (value->text == NULL) {
                json_free(value);
                value = NULL;
            }
        } else {
            set_error(&reader->error, "Memory allocation failed.");
        }
        break;
    case 't':
    case 'f': {
        const char *literal = reader->data[reader->position] == 't' ? "true" : "false";
        size_t length = strlen(literal);
        if (reader->length - reader->position < length ||
            memcmp(reader->data + reader->position, literal, length) != 0) {
            set_error(&reader->error, "Invalid Safetensors header JSON: malformed boolean.");
            value = NULL;
        } else {
            value = json_new(JSON_BOOL);
            if (value == NULL) set_error(&reader->error, "Memory allocation failed.");
            else value->boolean = literal[0] == 't';
            reader->position += length;
        }
        break;
    }
    case 'n':
        if (reader->length - reader->position < 4 ||
            memcmp(reader->data + reader->position, "null", 4) != 0) {
            set_error(&reader->error, "Invalid Safetensors header JSON: malformed null.");
            value = NULL;
        } else {
            value = json_new(JSON_NULL);
            if (value == NULL) set_error(&reader->error, "Memory allocation failed.");
            reader->position += 4;
        }
        break;
    default:
        if (reader->data[reader->position] == '-' ||
            (reader->data[reader->position] >= '0' && reader->data[reader->position] <= '9')) {
            value = json_parse_number(reader);
        } else {
            set_error(&reader->error, "Invalid Safetensors header JSON: unexpected character.");
            value = NULL;
        }
        break;
    }
    reader->depth--;
    return value;
}

static JsonValue *json_object_get(const JsonValue *object, const char *key) {
    size_t index;
    if (object == NULL || object->type != JSON_OBJECT) return NULL;
    for (index = 0; index < object->count; index++) {
        if (internal_string_compare(object->keys[index], key) == 0) return object->items[index];
    }
    return NULL;
}

static int json_u64(const JsonValue *value, uint64_t *number) {
    char *end = NULL;
    unsigned long long parsed;
    if (value == NULL || value->type != JSON_NUMBER || value->text[0] == '-') return 0;
    if (strchr(value->text, '.') != NULL || strchr(value->text, 'e') != NULL ||
        strchr(value->text, 'E') != NULL) return 0;
    errno = 0;
    parsed = strtoull(value->text, &end, 10);
    if (errno != 0 || end == value->text || *end != '\0') return 0;
    *number = (uint64_t)parsed;
    return 1;
}

static int compare_json_keys(const void *left, const void *right) {
    const char *const *a = (const char *const *)left;
    const char *const *b = (const char *const *)right;
    return internal_string_compare(*a, *b);
}

static int parse_safetensors(const uint8_t *data, size_t length, Model *model, char **error) {
    uint64_t header_length_u64;
    size_t header_length;
    JsonReader reader;
    JsonValue *root = NULL;
    JsonValue *metadata;
    char **keys = NULL;
    size_t key_count = 0;
    size_t index;
    size_t payload_length;
    Range *ranges = NULL;

    model->format = "Safetensors";
    if (length < 8) {
        set_error(error, "Safetensors file is missing the header length.");
        return 0;
    }
    header_length_u64 = 0;
    for (index = 0; index < 8; index++) {
        header_length_u64 |= (uint64_t)data[index] << (index * 8);
    }
    if (header_length_u64 > SIZE_MAX) {
        set_error(error, "Safetensors header length exceeds the range supported by this platform.");
        return 0;
    }
    header_length = (size_t)header_length_u64;
    if (header_length > MAX_HEADER_SIZE) {
        set_error(error, "Safetensors header is too large (%zu bytes; limit: %zu bytes).",
                  header_length, MAX_HEADER_SIZE);
        return 0;
    }
    if (header_length > length - 8) {
        set_error(error, "Safetensors file header is incomplete.");
        return 0;
    }
    reader = (JsonReader){data + 8, header_length, 0, 0, NULL};
    root = json_parse_value(&reader);
    json_skip_whitespace(&reader);
    if (root == NULL || reader.position != reader.length) {
        if (reader.error == NULL) set_error(&reader.error, "Invalid Safetensors header JSON.");
        *error = reader.error;
        json_free(root);
        return 0;
    }
    free(reader.error);
    if (root->type != JSON_OBJECT) {
        set_error(error, "Safetensors header must be a JSON object.");
        json_free(root);
        return 0;
    }
    payload_length = length - 8 - header_length;
    metadata = json_object_get(root, "__metadata__");
    if (metadata != NULL && metadata->type == JSON_OBJECT) {
        key_count = metadata->count;
        keys = (char **)calloc(key_count, sizeof(char *));
        if (key_count != 0 && keys == NULL) goto allocation_failure;
        for (index = 0; index < key_count; index++) keys[index] = metadata->keys[index];
        if (key_count > 1) qsort(keys, key_count, sizeof(char *), compare_json_keys);
        for (index = 0; index < key_count; index++) {
            JsonValue *value = json_object_get(metadata, keys[index]);
            const char *text = value->type == JSON_STRING ? value->text : "<non-string>";
            if (!model_add_metadata(model, keys[index], text)) goto allocation_failure;
        }
        free(keys);
        keys = NULL;
    }

    key_count = root->count;
    keys = (char **)calloc(key_count, sizeof(char *));
    if (key_count != 0 && keys == NULL) goto allocation_failure;
    {
        size_t tensor_keys = 0;
        for (index = 0; index < root->count; index++) {
            if (internal_string_compare(root->keys[index], "__metadata__") != 0) {
                keys[tensor_keys++] = root->keys[index];
            }
        }
        key_count = tensor_keys;
    }
    if (key_count > 1) qsort(keys, key_count, sizeof(char *), compare_json_keys);
    if (key_count != 0) {
        ranges = (Range *)calloc(key_count, sizeof(Range));
        if (ranges == NULL) goto allocation_failure;
    }
    for (index = 0; index < key_count; index++) {
        const char *name = keys[index];
        JsonValue *entry = json_object_get(root, name);
        JsonValue *dtype_value;
        JsonValue *shape_value;
        JsonValue *offsets_value;
        Tensor tensor = {0};
        uint64_t start;
        uint64_t end;
        size_t dimension;
        if (entry->type != JSON_OBJECT) {
            set_error(error, "Invalid description for tensor '%s'.", name);
            goto failed;
        }
        dtype_value = json_object_get(entry, "dtype");
        if (dtype_value == NULL || dtype_value->type != JSON_STRING) {
            set_error(error, "Tensor '%s' is missing dtype.", name);
            goto failed;
        }
        shape_value = json_object_get(entry, "shape");
        if (shape_value == NULL || shape_value->type != JSON_ARRAY) {
            set_error(error, "Tensor '%s' is missing a valid shape.", name);
            goto failed;
        }
        tensor.dimensions = shape_value->count;
        if (tensor.dimensions != 0) {
            if (tensor.dimensions > SIZE_MAX / sizeof(uint64_t)) goto allocation_failure;
            tensor.shape = (uint64_t *)malloc(tensor.dimensions * sizeof(uint64_t));
            if (tensor.shape == NULL) goto allocation_failure;
        }
        for (dimension = 0; dimension < tensor.dimensions; dimension++) {
            if (!json_u64(shape_value->items[dimension], &tensor.shape[dimension])) {
                set_error(error, "Tensor '%s' contains an invalid dimension.", name);
                free(tensor.shape);
                goto failed;
            }
        }
        offsets_value = json_object_get(entry, "data_offsets");
        if (offsets_value == NULL || offsets_value->type != JSON_ARRAY ||
            offsets_value->count != 2) {
            set_error(error, "Tensor '%s' is missing valid data_offsets.", name);
            free(tensor.shape);
            goto failed;
        }
        if (!json_u64(offsets_value->items[0], &start)) {
            set_error(error, "Invalid start offset for tensor '%s'.", name);
            free(tensor.shape);
            goto failed;
        }
        if (!json_u64(offsets_value->items[1], &end)) {
            set_error(error, "Invalid end offset for tensor '%s'.", name);
            free(tensor.shape);
            goto failed;
        }
        if (start > end || end > (uint64_t)payload_length) {
            set_error(error, "Data range %" PRIu64 "..%" PRIu64 " for tensor '%s' exceeds the file data section (%zu bytes).",
                      start, end, name, payload_length);
            free(tensor.shape);
            goto failed;
        }
        tensor.name = duplicate_string(name);
        tensor.dtype = duplicate_string(dtype_value->text);
        tensor.has_bytes = 1;
        tensor.bytes = end - start;
        if (tensor.name == NULL || tensor.dtype == NULL ||
            !model_add_tensor(model, &tensor)) {
            free(tensor.name);
            free(tensor.dtype);
            free(tensor.shape);
            goto allocation_failure;
        }
        ranges[index] = (Range){
            start, end, model->tensors[model->tensor_count - 1].name, index
        };
    }
    if (key_count > 1) qsort(ranges, key_count, sizeof(Range), compare_ranges);
    for (index = 1; index < key_count; index++) {
        if (ranges[index - 1].end > ranges[index].start) {
            set_error(error, "Data ranges for tensors '%s' and '%s' overlap.",
                      ranges[index - 1].name, ranges[index].name);
            goto failed;
        }
    }
    free(ranges);
    free(keys);
    json_free(root);
    return 1;

allocation_failure:
    set_error(error, "Memory allocation failed.");
failed:
    free(ranges);
    free(keys);
    json_free(root);
    return 0;
}

static int serialize_model(const Model *model, char **output) {
    Buffer buffer = {0};
    size_t index;
    size_t dimension;
    if (!buffer_append(&buffer, "{\"format\":") ||
        !buffer_append_json_string(&buffer, model->format) ||
        !buffer_append(&buffer, ",\"tensors\":[")) goto failed;
    for (index = 0; index < model->tensor_count; index++) {
        const Tensor *tensor = &model->tensors[index];
        if (index != 0 && !buffer_append_char(&buffer, ',')) goto failed;
        if (!buffer_append(&buffer, "{\"name\":") ||
            !buffer_append_json_string(&buffer, tensor->name) ||
            !buffer_append(&buffer, ",\"shape\":[")) goto failed;
        for (dimension = 0; dimension < tensor->dimensions; dimension++) {
            if (dimension != 0 && !buffer_append_char(&buffer, ',')) goto failed;
            if (!buffer_append_format(&buffer, "%" PRIu64, tensor->shape[dimension])) goto failed;
        }
        if (!buffer_append(&buffer, "],\"dtype\":") ||
            !buffer_append_json_string(&buffer, tensor->dtype) ||
            !buffer_append(&buffer, ",\"bytes\":")) goto failed;
        if (tensor->has_bytes) {
            if (!buffer_append_format(&buffer, "%" PRIu64, tensor->bytes)) goto failed;
        } else if (!buffer_append(&buffer, "null")) {
            goto failed;
        }
        if (!buffer_append_char(&buffer, '}')) goto failed;
    }
    if (!buffer_append(&buffer, "],\"metadata\":[")) goto failed;
    for (index = 0; index < model->metadata_count; index++) {
        if (index != 0 && !buffer_append_char(&buffer, ',')) goto failed;
        if (!buffer_append_char(&buffer, '[') ||
            !buffer_append_json_string(&buffer, model->metadata[index].key) ||
            !buffer_append_char(&buffer, ',') ||
            !buffer_append_json_string(&buffer, model->metadata[index].value) ||
            !buffer_append_char(&buffer, ']')) goto failed;
    }
    if (!buffer_append(&buffer, "]}")) goto failed;
    *output = buffer.data;
    return 1;
failed:
    free(buffer.data);
    return 0;
}

int wolf_parse_model(const uint8_t *data, size_t length, char **json_out, char **error_out) {
    Model model = {0};
    int parsed;
    *json_out = NULL;
    *error_out = NULL;
    if (data == NULL && length != 0) {
        *error_out = duplicate_string("Invalid model data pointer.");
        return 0;
    }
    if (length >= 4 && memcmp(data, "GGUF", 4) == 0) {
        parsed = parse_gguf(data, length, &model, error_out);
    } else if (length >= 8) {
        uint64_t header_length = 0;
        size_t index;
        for (index = 0; index < 8; index++) {
            header_length |= (uint64_t)data[index] << (index * 8);
        }
        if (header_length > (uint64_t)(length - 8)) {
            *error_out = duplicate_string(
                "Unable to identify file format; supported formats are GGUF and Safetensors (.gguf, .safetensors).");
            parsed = 0;
        } else {
            parsed = parse_safetensors(data, length, &model, error_out);
        }
    } else {
        *error_out = duplicate_string(
            "File is too short or empty; supported formats are GGUF and Safetensors (.gguf, .safetensors).");
        parsed = 0;
    }
    if (parsed && !serialize_model(&model, json_out)) {
        *error_out = duplicate_string("Memory allocation failed.");
        parsed = 0;
    }
    free_model(&model);
    return parsed;
}

void wolf_parser_free(void *pointer) {
    free(pointer);
}
