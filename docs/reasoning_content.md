# Reasoning Content Support

The InstructLab Training library supports structured reasoning traces through the `reasoning_content` field in message samples. This feature enables training models that can separate their thinking process from their final output.

## Overview

The `reasoning_content` field is an optional addition to the standard message format that allows you to include the model's internal reasoning process alongside the final response. This is particularly useful for:

- Training reasoning-capable models that show their work
- Supporting models that need to generate step-by-step reasoning
- Enabling chain-of-thought style training data
- Separating internal thinking from user-facing responses

## Message Format

### Standard Message Format

```json
{
  "role": "assistant",
  "content": "The answer is 42."
}
```

### Extended Message Format with Reasoning Content

```json
{
  "role": "assistant", 
  "content": "The answer is 42.",
  "reasoning_content": "Let me think about this step by step. The question asks for the meaning of life, and according to The Hitchhiker's Guide to the Galaxy, the answer is 42."
}
```

## Data Processing Behavior

When processing messages during training:

1. **Unmasking Rules**: Both `content` and `reasoning_content` fields follow the same unmasking rules based on the message role
2. **Template Integration**: Both fields are processed by the chat template and included in the tokenized output
3. **Token Wrapping**: If a role is configured to be unmasked, both fields (when present) are wrapped with unmask tokens
4. **Independent Fields**: Either field can exist independently - messages can have only `content`, only `reasoning_content`, or both

## Usage Examples

### Training Data with Reasoning Traces

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is 15 * 23?"
    },
    {
      "role": "assistant",
      "reasoning_content": "I need to multiply 15 by 23. Let me break this down: 15 * 23 = 15 * (20 + 3) = 15 * 20 + 15 * 3 = 300 + 45 = 345",
      "content": "15 * 23 = 345"
    }
  ]
}
```

### Mixed Content Types

```json
{
  "messages": [
    {
      "role": "user",
      "content": "Solve this math problem step by step: 2x + 5 = 13"
    },
    {
      "role": "assistant",
      "reasoning_content": "I need to solve for x. First, I'll subtract 5 from both sides: 2x = 8. Then divide by 2: x = 4.",
      "content": "To solve 2x + 5 = 13:\n1. Subtract 5 from both sides: 2x = 8\n2. Divide by 2: x = 4\n\nTherefore, x = 4."
    }
  ]
}
```

### Reasoning-Only Responses

```json
{
  "messages": [
    {
      "role": "user", 
      "content": "Think about the implications of AI safety."
    },
    {
      "role": "assistant",
      "reasoning_content": "This is a complex topic that requires careful consideration of multiple factors including alignment, capability control, and social implications..."
    }
  ]
}
```

## Implementation Details

### Token Processing

During data processing, the library:

1. Wraps both `content` and `reasoning_content` with special unmask tokens (`<|UNMASK_BEGIN|>`, `<|UNMASK_END|>`, `<|UNMASK_REASONING_BEGIN|>`, `<|UNMASK_REASONING_END|>`)
2. Applies the chat template to the combined message content
3. Processes the tokenized sequence to create appropriate labels for training
4. Removes the special unmask tokens from the final training data

### Validation

The library validates that:

- Both `content` and `reasoning_content` must be strings if present
- Special unmask tokens are properly processed and removed
- The final training data contains no residual unmask tokens

### Error Handling

Common errors and their meanings:

- `"unmasking non-string data types is currently unsupported"`: The `content` field contains non-string data
- `"received an entry for reasoning_content which was not a string"`: The `reasoning_content` field contains non-string data

## Integration with Existing Features

### Unmasking Policies

The `reasoning_content` field respects all existing unmasking policies:

- When `unmask=true` is set on a sample, both fields are unmasked for non-system roles
- When `unmask=false` (default), only assistant role messages are unmasked
- Custom unmask role configurations work with both fields

### Chat Templates

The `reasoning_content` is unsupported by the legacy chat templates and will not be rendered.

### Backward Compatibility

The feature is fully backward compatible:

- Existing datasets without `reasoning_content` continue to work unchanged
- All existing training configurations and arguments remain valid

## Testing

The library includes comprehensive tests for reasoning content functionality:

- Unit tests for message wrapping and processing
- Integration tests with real tokenizers
- Validation tests for error conditions
- Backward compatibility tests

## Important Notes

### Automatic Processing Behavior

1. **Always processed when present**: If `reasoning_content` exists in a message, it will always be processed and unmasked as long as the message role is targeted for unmasking. This ensures that reasoning traces are properly included in the training data without requiring additional configuration.

2. **DeepSeek R1 and Qwen3 compatibility**: Models using the DeepSeek R1 thought processor (such as Qwen3) **must** supply their thinking traces in the `reasoning_content` field to be processed correctly. Failure to do so may result in improper handling of reasoning tokens and suboptimal training performance.

3. **Separate token handling**: The library uses distinct unmask tokens for reasoning content (`<|UNMASK_REASONING_BEGIN|>` and `<|UNMASK_REASONING_END|>`) versus regular content (`<|UNMASK_BEGIN|>` and `<|UNMASK_END|>`), allowing for proper differentiation during training.

## Best Practices

1. **Consistent Usage**: When applicable, use `reasoning_content` consistently within a dataset for best results
2. **Clear Separation**: Keep reasoning traces separate from final outputs for clarity
3. **Template Compatibility**: Ensure your chat template properly handles both fields
4. **Validation**: Test your data processing pipeline with small samples before full training

## Migration Guide

To add reasoning content support to existing datasets:

1. Add `reasoning_content` fields to relevant messages
2. Ensure content is in string format
3. Test with a small sample using the data processing pipeline
4. Verify that unmask tokens are properly processed

No changes to training arguments or configuration are required.
