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
	set({ loading: true });
	const pendingId = Math.random();

	try {
    _addPendingMessage(input, pendingId);

    const conversationId = store.get().activeConversationId;

		// Get research mode state from store if not specified in opts
		const currentState = store.get();
		const useResearch = opts.useResearch !== undefined ? opts.useResearch : currentState.researchMode;
		const activeDocs = opts.activeDocs || currentState.activeDocuments.map(doc => doc.id);

		// Log research mode state for debugging
    console.log(`[sync] Sending message with research mode: ${useResearch}, activeDocs: ${activeDocs.length}`);

    // Include research mode parameters in the request
    const { data: responseMessage } = await api.post(
      `/conversations/${conversationId}/messages`,
      {
        input: input.content,
        useResearch: useResearch,
        activeDocs: activeDocs,
        useStreaming: false
      }
    );

		removeMessageFromActive(pendingId);
		insertMessageToActive(responseMessage);

		// Check if response has research mode metadata and update conversation if needed
		if (responseMessage.metadata?.research_mode) {
			console.log("[sync] Response includes research mode metadata:", responseMessage.metadata.research_mode);
		}

		set({ error: '', loading: false });
	} catch (err) {
		set({ error: getErrorMessage(err), loading: false });
	}
};
