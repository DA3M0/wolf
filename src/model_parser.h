#ifndef WOLF_MODEL_PARSER_H
#define WOLF_MODEL_PARSER_H

#include <stddef.h>
#include <stdint.h>

int wolf_parse_model(const uint8_t *data, size_t length, char **json_out, char **error_out);
void wolf_parser_free(void *pointer);

#endif
