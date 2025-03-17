import type { Message, MessageOpts } from './store';
import { store, set, insertMessageToActive, removeMessageFromActive } from './store';
import { api, getErrorMessage } from '$api';

const _addPendingMessage = (message: Message, pendingId: number) => {
	insertMessageToActive(message);
	insertMessageToActive({
		id: pendingId,
		role: 'pending',
		content: '...'
	});
};

export const sendMessage = async (input: Message, opts: MessageOpts = {}) => {
  set({ loading: true, error: '', lastMessageFailed: false });
  const pendingId = Math.random();

  try {
    _addPendingMessage(input, pendingId);

    const conversationId = store.get().activeConversationId;
    if (!conversationId) {
      throw new Error("No active conversation");
    }

    // Get research mode state from store if not specified in opts
    const currentState = store.get();
    const useResearch = opts.useResearch !== undefined ? opts.useResearch : currentState.researchMode;
    const activeDocs = opts.activeDocs || currentState.activeDocuments.map(doc => doc.id);

    console.log(`[sync] Sending message with research mode: ${useResearch}, activeDocs: ${activeDocs.length}`);

    // Include research mode parameters in the request
    const { data: responseMessage } = await api.post(
      `/conversations/${conversationId}/messages`,
      {
        input: input.content,
        message: input.content, // Add this for backward compatibility
        useResearch: useResearch,
        activeDocs: activeDocs,
        useStreaming: false
      }
    );

    removeMessageFromActive(pendingId);

    if (responseMessage) {
      insertMessageToActive(responseMessage);
    } else {
      throw new Error("Invalid response from server");
    }

    set({ error: '', loading: false, lastMessageFailed: false });
    return responseMessage;
  } catch (err) {
    removeMessageFromActive(pendingId);

    console.error("Error sending message:", err);
    set({
      error: getErrorMessage(err),
      loading: false,
      lastMessageFailed: true
    });

    return null;
  }
};
