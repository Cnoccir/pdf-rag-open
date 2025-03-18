/**
 * Utilities for parsing streaming responses in NDJSON format
 */

export interface StreamChunk {
  type: 'status' | 'stream' | 'end' | 'error';
  chunk?: string;
  message?: string;
  status?: string;
  error?: string;
  citations?: any[];
  conversation_id?: string;
  index?: number;
  is_complete?: boolean;
}

/**
 * Parse a line of NDJSON from the streaming response
 * @param line The text line to parse
 * @returns Parsed chunk or null if invalid
 */
export function parseStreamChunk(line: string): StreamChunk | null {
  if (!line || !line.trim()) return null;

  try {
    const chunk = JSON.parse(line) as StreamChunk;

    // Validate that it has required type property
    if (!chunk.type) {
      console.warn('Stream chunk missing type property:', chunk);
      return null;
    }

    return chunk;
  } catch (e) {
    console.warn('Failed to parse stream chunk:', line, e);
    // If it's not valid JSON, treat as a plain text chunk
    return {
      type: 'stream',
      chunk: line
    };
  }
}

/**
 * Process a chunk of streamed text that may contain multiple NDJSON lines
 * @param text The raw text chunk that may contain multiple lines
 * @returns Array of parsed chunks
 */
export function processStreamText(text: string): StreamChunk[] {
  const lines = text.split('\n').filter(line => line.trim());

  return lines
    .map(line => parseStreamChunk(line))
    .filter(chunk => chunk !== null) as StreamChunk[];
}
