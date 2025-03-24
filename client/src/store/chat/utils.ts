// Add this to a new file: client/src/store/chat/utils.ts

import { store } from './store';

/**
 * Safely update the store using immutable patterns
 * This helps prevent Immer errors by always creating new state objects
 */
export function safeStoreUpdate(updateFn: (state: any) => any) {
  try {
    store.update(currentState => {
      // Deep clone the current state to avoid Immer frozen state issues
      const newState = JSON.parse(JSON.stringify(currentState));

      // Apply the update function to the new state
      return updateFn(newState);
    });
    return true;
  } catch (error) {
    console.error("Safe store update failed:", error);
    return false;
  }
}

/**
 * Safely add a message to the active conversation
 */
export function safeAddMessage(message: any) {
  return safeStoreUpdate(state => {
    const conv = state.conversations.find(c => c.id === state.activeConversationId);
    if (conv) {
      if (!Array.isArray(conv.messages)) {
        conv.messages = [];
      }
      conv.messages.push({
        ...message,
        id: message.id || Date.now() + Math.random()
      });
    }
    return state;
  });
}

/**
 * Safely update an existing message in the active conversation
 */
export function safeUpdateMessage(id: string | number, updates: any) {
  return safeStoreUpdate(state => {
    const conv = state.conversations.find(c => c.id === state.activeConversationId);
    if (conv && Array.isArray(conv.messages)) {
      conv.messages = conv.messages.map(msg => {
        if (msg.id === id) {
          return { ...msg, ...updates };
        }
        return msg;
      });
    }
    return state;
  });
}

/**
 * Safely remove a message from the active conversation
 */
export function safeRemoveMessage(id: string | number) {
  return safeStoreUpdate(state => {
    const conv = state.conversations.find(c => c.id === state.activeConversationId);
    if (conv && Array.isArray(conv.messages)) {
      conv.messages = conv.messages.filter(msg => msg.id !== id);
    }
    return state;
  });
}
