import { get } from 'svelte/store';
import type { Message, MessageOpts } from './store';
import { set, store, getActiveConversation, insertMessageToActive, updateResearchModeFromActiveConversation } from './store';
import { addError } from '$s/errors';
import { getErrorMessage } from '$api';
import { processStreamText, type StreamChunk } from '../../utils/streamParser';

const _addMessage = (message: Message) => {
  insertMessageToActive(message);
};

const _appendResponse = (id: number, text: string) => {
  store.update((state) => {
    const conv = state.conversations.find((c) => c.id === state.activeConversationId);
    if (!conv) {
      return state;
    }
    conv.messages = conv.messages.map((message) => {
      if (message.id === id) {
        message.content += text;
        message.role = 'assistant';
      }
      return message;
    });
    return state;
  });
};

const _updateResponseWithMetadata = (id: number, content: string, metadata: any = {}) => {
  store.update((state) => {
    const conv = state.conversations.find((c) => c.id === state.activeConversationId);
    if (!conv) {
      return state;
    }
    conv.messages = conv.messages.map((message) => {
      if (message.id === id) {
        message.content = content;
        message.role = 'assistant';
        message.metadata = metadata;
      }
      return message;
    });
    return state;
  });
};

export const sendMessage = async (userMessage: Message, opts: MessageOpts = {}) => {
  const conversation = getActiveConversation();

  if (!conversation) {
    set({
      error: "No active conversation found. Please create a new chat.",
      loading: false,
      lastMessageFailed: true
    });
    return;
  }

  set({ loading: true, error: '', lastMessageFailed: false });

  const responseMessage = {
    role: 'pending',
    content: '',
    id: Math.random()
  } as Message;

  try {
    _addMessage(userMessage);
    _addMessage(responseMessage);

    // Get research mode state from store if not specified in opts
    const currentState = get(store);
    const useResearch = opts.useResearch !== undefined ? opts.useResearch : currentState.researchMode;
    const activeDocs = opts.activeDocs || currentState.activeDocuments.map((doc) => doc.id);

    console.log(`[stream] Sending message with research mode: ${useResearch}, activeDocs: ${activeDocs.length}`);

    // Include research mode parameters in the request
    const response = await fetch(`/api/conversations/${conversation.id}/messages`, {
      method: 'POST',
      body: JSON.stringify({
        input: userMessage.content,
        message: userMessage.content, // Add for backward compatibility
        useResearch: useResearch,
        activeDocs: activeDocs,
        useStreaming: true
      }),
      credentials: 'include',
      headers: {
        'Content-Type': 'application/json'
      }
    });

    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error("Stream not available");
    }

    if (response.status >= 400) {
      await readError(response.status, reader);
    } else {
      await readResponse(reader, responseMessage);

      // Update research mode status after message completes
      updateResearchModeFromActiveConversation();
    }
  } catch (err) {
    set({
      error: getErrorMessage(err),
      loading: false,
      lastMessageFailed: true
    });
  }
};

const readResponse = async (
  reader: ReadableStreamDefaultReader<Uint8Array>,
  responseMessage: Message
) => {
  let inProgress = true;
  let completeMessage = '';
  let citations = [];
  let conversationId = null;

  while (inProgress) {
    try {
      const { done, value } = await reader.read();
      if (done) {
        inProgress = false;
        break;
      }

      const text = new TextDecoder().decode(value);
      const chunks = processStreamText(text);

      for (const chunk of chunks) {
        switch (chunk.type) {
          case 'status':
            // Just log status updates
            console.log(`[Stream Status] ${chunk.status}: ${chunk.message}`);
            break;

          case 'stream':
            // Append to the message content
            if (responseMessage.id && chunk.chunk) {
              _appendResponse(responseMessage.id, chunk.chunk);
              completeMessage += chunk.chunk;
            }
            break;

          case 'end':
            // Final message with full content
            if (responseMessage.id && chunk.message) {
              // Store metadata for later use
              completeMessage = chunk.message;
              citations = chunk.citations || [];
              conversationId = chunk.conversation_id;

              // Update the message with final content and metadata
              _updateResponseWithMetadata(
                responseMessage.id,
                completeMessage,
                {
                  citations,
                  conversation_id: conversationId
                }
              );
            }
            inProgress = false;
            break;

          case 'error':
            // Handle error messages
            if (chunk.error) {
              set({
                error: chunk.error,
                loading: false,
                lastMessageFailed: true
              });
              inProgress = false;
            }
            break;

          default:
            // Unknown chunk type, just log it
            console.warn('Unknown chunk type:', chunk);
        }
      }
    } catch (err) {
      console.error('Error reading stream:', err);
      inProgress = false;
    }
  }

  // Set loading to false when streaming completes
  set({ loading: false });
};

const readError = async (statusCode: number, reader: ReadableStreamDefaultReader<Uint8Array>) => {
  let inProgress = true;
  let message = '';
  while (inProgress) {
    const { done, value } = await reader.read();
    if (done) {
      inProgress = false;
      break;
    }
    const text = new TextDecoder().decode(value);
    message += text;
  }

  if (statusCode >= 500) {
    addError({ message, contentType: message.includes('<!doctype html>') ? 'text/html' : '' });
  } else {
    try {
      set({ error: getErrorMessage(JSON.parse(message)), loading: false });
    } catch (err) {
      set({ error: getErrorMessage(message), loading: false });
    }
  }
};
