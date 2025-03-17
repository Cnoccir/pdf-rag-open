import type { Message, MessageOpts, Conversation, DocumentInfo } from './store';
import {
    store,
    set,
    resetAll,
    resetError,
    fetchConversations,
    createConversation,
    setActiveConversationId,
    getActiveConversation,
    scoreConversation,
    activateResearchMode,
    deactivateResearchMode,
    acceptRecommendedDocument,
    fetchAvailableDocuments,
    sendMessage,
    regenerateResponse,
    deleteDocument  // Import correctly from store.ts
} from './store';
import { sendMessage as sendStreamingMessage } from './stream';
import { sendMessage as sendSyncMessage } from './sync';

// Define the function to choose between streaming and non-streaming message sending
const sendMessageWithMode = (message: Message, opts: MessageOpts) => {
    return opts.useStreaming ? sendStreamingMessage(message, opts) : sendSyncMessage(message, opts);
};

export {
    store,
    set,
    sendMessage,
    resetAll,
    resetError,
    fetchConversations,
    createConversation,
    setActiveConversationId,
    getActiveConversation,
    Conversation,
    scoreConversation,
    activateResearchMode,
    deactivateResearchMode,
    acceptRecommendedDocument,
    fetchAvailableDocuments,
    DocumentInfo,
    regenerateResponse,
    deleteDocument,
    sendMessageWithMode  // Export this utility function
};
