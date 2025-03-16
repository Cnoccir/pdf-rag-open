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
    regenerateResponse  // Import this from store.js
} from './store.js';
import { sendMessage as sendStreamingMessage } from './stream';
import { sendMessage as sendSyncMessage } from './sync';

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
    regenerateResponse,  // Export regenerateResponse
    deleteDocument
};
