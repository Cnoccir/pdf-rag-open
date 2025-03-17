import { get } from 'svelte/store';
import type { Message, MessageOpts } from './store';
import { set, store, getActiveConversation, insertMessageToActive } from './store';
import { addError } from '$s/errors';
import { getErrorMessage } from '$api';

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
    const response = await fetch(`/api/conversations/${conversation.id}/messages?stream=true`, {
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

	while (inProgress) {
		const { done, value } = await reader.read();
		if (done) {
			inProgress = false;
			break;
		}
		const text = new TextDecoder().decode(value);

		if (responseMessage.id) {
			_appendResponse(responseMessage.id, text);
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
