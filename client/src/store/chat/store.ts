import { get } from 'svelte/store'; // Ensure get is imported
import { writable } from '../writeable';
import { api } from '$api';

// Enable for debugging
const DEBUG = true;

// Helper logging function
const logDebug = (message: string, data: any = null) => {
  if (!DEBUG) return;
  if (data) {
    console.log(`[Chat Debug] ${message}:`, data);
  } else {
    console.log(`[Chat Debug] ${message}`);
  }
};

// Type definitions for conversation and messages
export interface Message {
  id?: string | number;
  role: 'user' | 'assistant' | 'system' | 'pending';
  content: string;
  metadata?: any;
  created_at?: string;
}

export interface Conversation {
  id: string;
  title: string;
  pdf_id?: string;
  messages: Message[];
  metadata?: {
    research_mode?: {
      active: boolean;
      pdf_ids: string[];
      document_names?: Record<string, string>;
    };
    [key: string]: any;
  };
  last_updated?: string;
  message_count?: number;
}

// Document interfaces for research mode
export interface DocumentInfo {
  id: string;
  name: string;
  concepts?: string[];
  confidence?: number;
  isPrimary?: boolean;
}

export interface MessageOpts {
  useStreaming?: boolean;
  useResearch?: boolean;
  documentId?: string;
  activeDocs?: string[];
}

// Main chat state interface
export interface ChatState {
  error: string;
  loading: boolean;
  activeConversationId: string | null;
  conversations: Conversation[];
  // Research mode fields
  researchMode: boolean;
  recommendedDocuments: DocumentInfo[];
  activeDocuments: DocumentInfo[];
  availableDocuments: DocumentInfo[];
  lastMessageFailed: boolean;
  streamInProgress: boolean;
}

// Initial state
const INITIAL_STATE: ChatState = {
  error: '',
  loading: false,
  activeConversationId: null,
  conversations: [],
  // Research mode initial state
  researchMode: false,
  recommendedDocuments: [],
  activeDocuments: [],
  availableDocuments: [],
  lastMessageFailed: false,
  streamInProgress: false
};

// Create store
const store = writable<ChatState>(INITIAL_STATE);

// Helper to update state
const set = (val: Partial<ChatState>) => {
  store.update((state) => ({ ...state, ...val }));
};

// Get active conversation
const getActiveConversation = (): Conversation | null => {
  const { conversations, activeConversationId } = get(store); // Fixed: using get(store)
  if (!activeConversationId) {
    return null;
  }

  return conversations.find((c) => c.id === activeConversationId) || null;
};

// Get raw messages (for API requests)
const getRawMessages = () => {
  const conversation = getActiveConversation();
  if (!conversation) {
    return [];
  }

  return conversation.messages
    .filter((message) => message.role !== 'pending')
    .map((message) => {
      return { role: message.role, content: message.content };
    });
};

// Insert message to active conversation
const insertMessageToActive = (message: Message) => {
  store.update((s) => {
    const conv = s.conversations.find((c) => c.id === s.activeConversationId);
    if (!conv) {
      return s;
    }

    // Ensure messages array is initialized
    if (!Array.isArray(conv.messages)) {
      conv.messages = [];
    }

    conv.messages.push(message);
    return s;
  });
};

// Remove message from active conversation
const removeMessageFromActive = (id: string | number) => {
  store.update((s) => {
    const conv = s.conversations.find((c) => c.id === s.activeConversationId);
    if (!conv) {
      return s;
    }

    // Ensure messages array is initialized
    if (!Array.isArray(conv.messages)) {
      conv.messages = [];
      return s;
    }

    conv.messages = conv.messages.filter((m) => m.id !== id);
    return s;
  });
};

// Add conversation scoring
const scoreConversation = async (score: number) => {
  const conversationId = get(store).activeConversationId; // Fixed: using get(store)
  if (!conversationId) return;

  try {
    await api.post(`/scores?conversation_id=${conversationId}`, { score });
    return true;
  } catch (error) {
    console.error("Error scoring conversation:", error);
    return false;
  }
};

// Extract recommended documents from the latest assistant message
const extractRecommendedDocuments = () => {
  const conversation = getActiveConversation();
  if (!conversation) return;

  // Find the last assistant message with recommendations
  const assistantMessages = conversation.messages.filter(
    m => m.role === 'assistant' && m.metadata?.recommended_documents
  );

  if (assistantMessages.length > 0) {
    const lastMessage = assistantMessages[assistantMessages.length - 1];
    const recommendedDocs = lastMessage.metadata?.recommended_documents || [];

    // Update recommended documents in store
    set({
      recommendedDocuments: recommendedDocs
    });

    logDebug("Updated recommended documents:", recommendedDocs);
  }
};

// Fetch conversations for a document
const fetchConversations = async (documentId: string) => {
  set({ loading: true, error: '', lastMessageFailed: false });

  try {
    logDebug(`Fetching conversations for document: ${documentId}`);
    const { data } = await api.get(`/conversations/${documentId}`);
    logDebug("Conversation API response:", data);

    // Extract conversations from response
    const processedConversations = data.conversations || [];
    logDebug(`Received ${processedConversations.length} conversations`);

    // Ensure each conversation has initialized messages array
    processedConversations.forEach(conv => {
      if (!Array.isArray(conv.messages)) {
        conv.messages = [];
      }
    });

    if (processedConversations.length) {
      set({
        conversations: processedConversations,
        activeConversationId: processedConversations[0].id,
        loading: false
      });

      // Set research mode if the active conversation has it enabled
      updateResearchModeFromActiveConversation();
    } else {
      await createConversation(documentId);
    }

    return processedConversations;
  } catch (error) {
    console.error("Error fetching conversations:", error);
    set({
      error: error.response?.data?.error || 'Failed to load conversations',
      loading: false,
      lastMessageFailed: true
    });

    return [];
  }
}

// Create new conversation
const createConversation = async (documentId: string) => {
  set({ loading: true, error: '', lastMessageFailed: false });

  try {
    logDebug(`Creating conversation for document: ${documentId}`);
    const { data } = await api.post(`/conversations/${documentId}`);
    logDebug("Create conversation API response:", data);

    // Ensure data has required structure
    if (!data || !data.id) {
      throw new Error("Invalid response from server");
    }

    // Ensure messages array is initialized
    if (!Array.isArray(data.messages)) {
      data.messages = [];
    }

    // Add to conversations list
    const currentConversations = get(store).conversations; // Fixed: using get(store)

    set({
      activeConversationId: data.id,
      conversations: [data, ...currentConversations],
      loading: false,
      // Reset research mode for new conversation
      researchMode: false,
      recommendedDocuments: [],
      activeDocuments: []
    });

    return data;
  } catch (error) {
    console.error("Error creating conversation:", error);
    set({
      error: error.response?.data?.error || 'Failed to create conversation',
      loading: false,
      lastMessageFailed: true
    });

    return null;
  }
}

// Set active conversation ID
const setActiveConversationId = (id: string) => {
  set({ activeConversationId: id, lastMessageFailed: false, error: '' });

  // Update research mode state based on the newly selected conversation
  updateResearchModeFromActiveConversation();
};

// Update research mode from active conversation metadata
const updateResearchModeFromActiveConversation = () => {
  const conversation = getActiveConversation();
  if (!conversation) {
    logDebug("No active conversation found for research mode update");
    return;
  }

  logDebug("Updating research mode from conversation:", conversation);

  // Check if conversation has research mode metadata
  if (conversation.metadata && conversation.metadata.research_mode) {
    const researchMode = conversation.metadata.research_mode;
    logDebug("Found research mode metadata:", researchMode);

    // Check both active flag AND multiple PDFs
    let isResearchActive = false;

    if (researchMode.active === true && researchMode.pdf_ids && researchMode.pdf_ids.length > 1) {
      isResearchActive = true;

      // Extract document information
      let activeDocuments = researchMode.pdf_ids
        .filter(id => id) // Filter out null/undefined IDs
        .map(id => {
          // Get document name from metadata if available
          const name = researchMode.document_names?.[id] || `Document ${id}`;

          return {
            id: id.toString(),
            name: name,
            isPrimary: String(id) === String(conversation.pdf_id)
          };
        })
        .filter(doc => doc !== null); // Remove invalid documents

      logDebug(`Setting research mode to ${isResearchActive} with ${activeDocuments.length} documents`);

      set({
        researchMode: isResearchActive,
        activeDocuments
      });
    } else {
      logDebug("Research mode not active or insufficient documents");
      set({
        researchMode: false,
        activeDocuments: []
      });
    }
  } else {
    logDebug("No research_mode metadata found");
    set({
      researchMode: false,
      activeDocuments: []
    });
  }
};

// Send a message
const sendMessage = async (text: string, opts: MessageOpts = {}) => {
  const conversationId = get(store).activeConversationId; // Fixed: using get(store)
  logDebug(`Sending message to conversation ${conversationId}, text: "${text.substring(0, 30)}..."`);

  if (!conversationId) {
    const errorMessage = "No active conversation. Please create a new chat first.";
    logDebug(errorMessage);
    set({
      error: errorMessage,
      lastMessageFailed: true
    });
    return null;
  }

  set({ loading: true, error: '', lastMessageFailed: false });

  // Get current research mode state
  const currentState = get(store); // Fixed: using get(store)
  const isResearchActive = currentState.researchMode;
  const activeDocuments = currentState.activeDocuments.map(doc => doc.id);

  logDebug(`Research mode: ${isResearchActive}, active docs:`, activeDocuments);

  // Decide between streaming and non-streaming
  const useStreaming = opts.useStreaming !== undefined ? opts.useStreaming : true;

  if (useStreaming) {
    // Use streaming API
    return streamMessage(text, conversationId, isResearchActive, activeDocuments);
  } else {
    // Use regular API
    return sendNonStreamingMessage(text, conversationId, isResearchActive, activeDocuments);
  }
};

// Send a message with streaming
const streamMessage = async (text: string, conversationId: string, isResearchActive: boolean, activeDocuments: string[]) => {
  // Add user message and pending message immediately for better UX
  const userMessage: Message = {
    id: Date.now(),
    role: 'user',
    content: text
  };

  const pendingMessage: Message = {
    id: Date.now() + 1,
    role: 'pending',
    content: ''
  };

  // Update store
  store.update(s => {
    const conversation = s.conversations.find(c => c.id === conversationId);
    if (conversation) {
      if (!Array.isArray(conversation.messages)) {
        conversation.messages = [];
      }
      conversation.messages.push(userMessage, pendingMessage);
    }
    return {
      ...s,
      streamInProgress: true
    };
  });

  try {
    // Use fetch for streaming support
    const response = await fetch(`/api/conversations/${conversationId}/messages`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        input: text,
        message: text, // For backwards compatibility
        useStreaming: true,
        useResearch: isResearchActive,
        activeDocs: activeDocuments
      })
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.error || `Server error: ${response.status}`);
    }

    // Get reader for the stream
    const reader = response.body?.getReader();
    if (!reader) {
      throw new Error("Stream not available");
    }

    let accumulatedText = '';
    let citationsData = [];
    let decoder = new TextDecoder();

    // Remove pending message and replace with actual assistant message
    store.update(s => {
      const conversation = s.conversations.find(c => c.id === conversationId);
      if (conversation) {
        conversation.messages = conversation.messages.filter(m => m.role !== 'pending');
        conversation.messages.push({
          id: Date.now() + 2,
          role: 'assistant',
          content: ''
        });
      }
      return s;
    });

    // Read the stream
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      // Decode the chunk
      const chunk = decoder.decode(value);

      // Process multiple JSON objects that might be in the same chunk
      const lines = chunk.split('\n').filter(line => line.trim());

      for (const line of lines) {
        try {
          const data = JSON.parse(line);

          if (data.type === 'status' && data.status === 'processing') {
            // Status update, do nothing yet
            continue;
          }

          if (data.type === 'error') {
            throw new Error(data.error || 'Unknown error in stream');
          }

          if (data.type === 'stream' && data.chunk) {
            // Append to accumulated text
            accumulatedText += data.chunk;

            // Update the message in real-time
            store.update(s => {
              const conversation = s.conversations.find(c => c.id === conversationId);
              if (conversation) {
                const assistantMsgIndex = conversation.messages.findIndex(m => m.role === 'assistant');
                if (assistantMsgIndex >= 0) {
                  conversation.messages[assistantMsgIndex].content = accumulatedText;
                }
              }
              return s;
            });
          }

          if (data.type === 'end') {
            // Final message with complete response
            accumulatedText = data.message || accumulatedText;
            citationsData = data.citations || [];

            // Update the final message with complete content and metadata
            store.update(s => {
              const conversation = s.conversations.find(c => c.id === conversationId);
              if (conversation) {
                const assistantMsgIndex = conversation.messages.findIndex(m => m.role === 'assistant');
                if (assistantMsgIndex >= 0) {
                  conversation.messages[assistantMsgIndex].content = accumulatedText;
                  conversation.messages[assistantMsgIndex].metadata = {
                    citations: citationsData
                  };
                }
              }
              return {
                ...s,
                loading: false,
                streamInProgress: false,
                lastMessageFailed: false
              };
            });

            // Update research mode state after receiving response
            updateResearchModeFromActiveConversation();

            return {
              content: accumulatedText,
              citations: citationsData
            };
          }
        } catch (jsonError) {
          console.error("Error parsing JSON from stream:", jsonError);
          // Continue to next line
        }
      }
    }

    // Ensure loading is set to false even if no 'end' message
    set({ loading: false, streamInProgress: false });
    return {
      content: accumulatedText,
      citations: citationsData
    };
  } catch (error) {
    console.error("Error in streaming message:", error);

    // Remove pending message if still present
    store.update(s => {
      const conversation = s.conversations.find(c => c.id === conversationId);
      if (conversation) {
        conversation.messages = conversation.messages.filter(m => m.role !== 'pending');
      }
      return {
        ...s,
        error: error.message || 'Failed to send message',
        loading: false,
        streamInProgress: false,
        lastMessageFailed: true
      };
    });

    return null;
  }
};

// Send a message without streaming
const sendNonStreamingMessage = async (text: string, conversationId: string, isResearchActive: boolean, activeDocuments: string[]) => {
  // Add user message immediately for better UX
  const userMessage: Message = {
    id: Date.now(),
    role: 'user',
    content: text
  };

  insertMessageToActive(userMessage);

  try {
    const response = await api.post(`/conversations/${conversationId}/messages`, {
      input: text,
      message: text, // For backwards compatibility
      useStreaming: false,
      useResearch: isResearchActive,
      activeDocs: activeDocuments
    });

    // Add the response from the API
    if (response.data && response.data.response) {
      const assistantMessage: Message = {
        id: Date.now() + 1,
        role: 'assistant',
        content: response.data.response,
        metadata: {
          citations: response.data.citations || []
        }
      };

      insertMessageToActive(assistantMessage);
    }

    // Update research mode state after receiving response
    updateResearchModeFromActiveConversation();

    set({ loading: false, lastMessageFailed: false });
    return response.data;
  } catch (error) {
    console.error("Error sending message:", error);
    set({
      error: error.response?.data?.error || error.message || 'Failed to send message',
      loading: false,
      lastMessageFailed: true
    });
    return null;
  }
};

// Regenerate the response for a previously failed query
const regenerateResponse = async (query: string, options: MessageOpts = {}) => {
  const currentState = get(store); // Fixed: using get(store)
  const conversationId = currentState.activeConversationId;

  if (!conversationId) {
    set({
      error: 'No active conversation. Please start a new chat or select an existing conversation.',
      loading: false,
      lastMessageFailed: true,
    });
    throw new Error('No active conversation');
  }

  set({ loading: true, error: '', lastMessageFailed: false });

  try {
    // Capture current research mode state
    const isResearchActive = currentState.researchMode;
    const activeDocuments = currentState.activeDocuments.map((doc) => doc.id);

    console.log(
      `Regenerating response with research mode: ${isResearchActive}, docs: ${activeDocuments.length}`
    );

    // Use the streaming approach for better UX
    return await streamMessage(query, conversationId, isResearchActive, activeDocuments);
  } catch (error) {
    console.error('Failed to regenerate response:', error);

    set({
      loading: false,
      error: error.response?.data?.error || 'Failed to regenerate response',
      lastMessageFailed: true,
    });

    throw error;
  }
};

// Activate research mode
const activateResearchMode = async (conversationId: number, pdfIds: string[]) => {
  set({ loading: true, error: '', lastMessageFailed: false });

  try {
    // Ensure all IDs are strings and filter out empty values
    const formattedPdfIds = (pdfIds || [])
      .map(id => id.toString())
      .filter(id => id && id.trim().length > 0);

    if (formattedPdfIds.length < 2) {
      throw new Error("At least two valid document IDs are required for research mode");
    }

    // Get current conversation to identify primary document
    const currentState = get(store); // Fixed: using get(store)
    const activeConv = currentState.conversations.find(c => c.id === conversationId);
    const primaryDocId = activeConv?.pdf_id?.toString();

    logDebug(`Primary document: ${primaryDocId}`);

    // Ensure primary document is included
    if (primaryDocId && !formattedPdfIds.includes(primaryDocId)) {
      formattedPdfIds.unshift(primaryDocId);
    }

    // Get document names
    const documentsWithNames = await fetchDocumentNames(formattedPdfIds);

    logDebug("Enhanced documents for research:", documentsWithNames);

    // FIXED: Use the correct API endpoint
    const response = await api.post(`/api/conversations/${conversationId}/research/activate`, {
      pdf_ids: formattedPdfIds
    });

    // Update store with properly named documents
    set({
      researchMode: true,
      activeDocuments: documentsWithNames,
      loading: false
    });

    // Update conversation metadata
    store.update(s => {
      const conversation = s.conversations.find(c => c.id === conversationId);
      if (conversation) {
        conversation.metadata = {
          ...conversation.metadata,
          research_mode: {
            active: true,
            pdf_ids: [...formattedPdfIds],
            document_names: documentsWithNames.reduce((acc, doc) => {
              acc[doc.id] = doc.name;
              return acc;
            }, {})
          }
        };
      }
      return s;
    });

    return true;
  } catch (error) {
    console.error("Error activating research mode:", error);
    set({
      error: error.response?.data?.error || 'Failed to activate research mode',
      loading: false,
      lastMessageFailed: true
    });
    return false;
  }
}

const deactivateResearchMode = async (conversationId: number) => {
  logDebug(`Deactivating research mode for conversation ${conversationId}`);
  set({ loading: true, error: '', lastMessageFailed: false });

  try {
    // FIXED: Use the correct API endpoint for toggling research mode
    const response = await api.post(`/api/conversations/${conversationId}/research/deactivate`, {
      active: false  // Explicitly set active flag to false
    });

    logDebug("Research mode deactivation response:", response.data);

    set({
      researchMode: false,
      activeDocuments: [],
      loading: false
    });

    // Update conversation metadata
    store.update(s => {
      const conversation = s.conversations.find(c => c.id === conversationId);
      if (conversation) {
        // Create a new metadata object to ensure state updates properly
        conversation.metadata = {
          ...conversation.metadata,
          research_mode: {
            active: false,
            pdf_ids: [],
            document_names: {}
          }
        };
      }
      return s;
    });

    return true;
  } catch (error) {
    const errorMessage = error.response?.data?.error || error.message || 'Failed to deactivate research mode';
    logDebug(`Error deactivating research mode: ${errorMessage}`, error);

    set({
      error: errorMessage,
      loading: false,
      lastMessageFailed: true
    });

    return false;
  }
};

// Accept a recommended document
const acceptRecommendedDocument = async (documentId: string) => {
  const state = get(store); // Fixed: using get(store)
  const conversationId = state.activeConversationId;

  if (!conversationId) return false;

  // Create list of all document IDs (existing + new)
  const currentDocIds = state.activeDocuments.map(d => d.id);
  const allDocIds = [...new Set([...currentDocIds, documentId])];

  // Update locally first for better UX
  store.update(s => ({
    ...s,
    recommendedDocuments: s.recommendedDocuments.filter(d => d.id !== documentId),
    loading: true
  }));

  logDebug(`Accepting recommended document ${documentId}, activating research mode with docs:`, allDocIds);

  // Activate research mode with the new document
  const success = await activateResearchMode(conversationId, allDocIds);

  // Ensure the UI state is updated correctly by re-reading from conversation metadata
  if (success) {
    updateResearchModeFromActiveConversation();
  }

  return success;
};

// Fetch available documents
const fetchAvailableDocuments = async () => {
  try {
    const { data } = await api.get('/pdfs/');

    // Format documents with proper interface
    const formattedDocs: DocumentInfo[] = data.map(doc => ({
      id: doc.id.toString(),
      name: doc.name || `Document ${doc.id}`
    }));

    // Store for future use
    set({ availableDocuments: formattedDocs });

    logDebug(`Fetched ${formattedDocs.length} available documents`);
    return formattedDocs;
  } catch (error) {
    console.error("Failed to fetch available documents:", error);
    return [];
  }
};

// Delete a document
const deleteDocument = async (documentId: string) => {
  try {
    await api.delete(`/pdfs/${documentId}`);
    logDebug(`Document ${documentId} deleted successfully`);

    // Update available documents list
    store.update(s => ({
      ...s,
      availableDocuments: s.availableDocuments.filter(doc => doc.id !== documentId)
    }));

    return true;
  } catch (error) {
    console.error("Error deleting document:", error);
    set({
      error: error.response?.data?.error || 'Failed to delete document',
      lastMessageFailed: true
    });
    return false;
  }
};

// Fetch document names
const fetchDocumentNames = async (pdfIds: string[]): Promise<DocumentInfo[]> => {
  try {
    // First, try to get names from existing documents in the store
    const existingDocs = get(store).availableDocuments; // Fixed: using get(store)
    const cachedConversation = getActiveConversation();
    const primaryPdf = cachedConversation?.pdf_id;

    logDebug("Fetching document names for:", pdfIds);
    logDebug("Primary PDF:", primaryPdf);
    logDebug("Available documents cache:", existingDocs);

    // If we have cached documents, use those names first
    if (existingDocs.length > 0) {
      return pdfIds.map(id => {
        // Check if this is the primary document from conversation
        if (primaryPdf && String(primaryPdf) === String(id)) {
          return {
            id: id.toString(),
            name: existingDocs.find(d => String(d.id) === String(id))?.name || `Document ${id}`,
            isPrimary: true
          };
        }

        // Otherwise look in cached documents
        const doc = existingDocs.find(d => String(d.id) === String(id));
        return {
          id: id.toString(),
          name: doc?.name || `Document ${id}`,
          isPrimary: false
        };
      });
    }

    // If no cached documents, fetch them individually
    logDebug("No cached documents, fetching individually");
    const docPromises = pdfIds.map(async id => {
      try {
        const { data } = await api.get(`/pdfs/${id}`);
        logDebug(`Fetched document ${id}:`, data);

        // Handle different response formats
        let name;
        if (data.pdf) {
          name = data.pdf.name || `Document ${id}`;
        } else if (data.name) {
          name = data.name;
        } else {
          name = `Document ${id}`;
        }

        return {
          id: id.toString(),
          name: name,
          isPrimary: primaryPdf ? String(primaryPdf) === String(id) : false
        };
      } catch (error) {
        console.error(`Failed to fetch document ${id}:`, error);
        return {
          id: id.toString(),
          name: `Document ${id}`,
          isPrimary: primaryPdf ? String(primaryPdf) === String(id) : false
        };
      }
    });

    const results = await Promise.all(docPromises);
    logDebug("Fetched document names:", results);
    return results;
  } catch (error) {
    console.error("Error fetching document names:", error);
    return pdfIds.map(id => ({
      id: id.toString(),
      name: `Document ${id}`,
      isPrimary: false
    }));
  }
};

// Reset store state
const resetAll = () => {
  set(INITIAL_STATE);
};

// Reset error state
const resetError = () => {
  set({ error: '', lastMessageFailed: false });
};

export {
  store,
  set,
  setActiveConversationId,
  getRawMessages,
  fetchConversations,
  resetAll,
  resetError,
  createConversation,
  getActiveConversation,
  insertMessageToActive,
  removeMessageFromActive,
  scoreConversation,
  sendMessage,
  regenerateResponse,
  activateResearchMode,
  deactivateResearchMode,
  acceptRecommendedDocument,
  fetchAvailableDocuments,
  updateResearchModeFromActiveConversation,
  deleteDocument,
  streamMessage
};
