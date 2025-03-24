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
    deleteDocument
} from './store';
import { sendMessage as sendStreamingMessage } from './stream';
import { sendMessage as sendSyncMessage } from './sync';

// Define the function to choose between streaming and non-streaming message sending
const sendMessageWithMode = (message: Message, opts: MessageOpts = {}) => {
    // Ensure we're using a properly formatted message object
    const formattedMessage: Message = {
        role: message.role || 'user',
        content: message.content || '',
        ...(message.metadata ? { metadata: message.metadata } : {})
    };

    // Ensure we have valid options with defaults
    const formattedOpts: MessageOpts = {
        useStreaming: true,  // Default to streaming
        ...opts
    };

    // Choose the appropriate sending method
    return formattedOpts.useStreaming
        ? sendStreamingMessage(formattedMessage, formattedOpts)
        : sendSyncMessage(formattedMessage, formattedOpts);
};

// Create a safe version of sendMessage that catches and logs errors
const safeSendMessage = (text: string, opts: MessageOpts = {}) => {
    try {
        // Create a proper message object
        const message: Message = {
            role: 'user',
            content: text
        };

        // Call the sendMessageWithMode function
        return sendMessageWithMode(message, opts);
    } catch (error) {
        console.error("Error in safeSendMessage:", error);
        // Update the store with the error
        set({
            error: error.message || "An error occurred while sending your message",
            loading: false,
            lastMessageFailed: true
        });
        return null;
    }
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
    sendMessageWithMode,  // Export the new function
    safeSendMessage      // Export the safe version
};
