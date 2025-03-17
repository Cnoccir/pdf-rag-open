import { get } from 'svelte/store';
import { writable } from '../writeable';
import { api } from '$api';

export interface Message {
	id?: number;
	role: 'user' | 'assistant' | 'system' | 'pending';
	content: string;
	metadata?: any; // Include metadata for research mode info
}

export interface Conversation {
	id: number;
	messages: Message[];
	pdf_id?: number; // Document ID
	metadata?: {
		research_mode?: {
			active: boolean;
			pdf_ids: string[];
			document_names?: Record<string, string>;
		}
	};
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
	activeDocs?: string[]; // Add active document IDs
}

export interface ChatState {
	error: string;
	loading: boolean;
	activeConversationId: number | null;
	conversations: Conversation[];
	// Research mode fields
	researchMode: boolean;
	recommendedDocuments: DocumentInfo[];
	activeDocuments: DocumentInfo[];
	availableDocuments: DocumentInfo[];
	// Add this new field to track failed messages
	lastMessageFailed: boolean;
}

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
	// Initialize lastMessageFailed
	lastMessageFailed: false
};

// Add DEBUG flag
const DEBUG = true;

// Define the logDebug function
const logDebug = (message: string, data: any = null) => {
  if (!DEBUG) return;

  if (data) {
    console.log(`[Chat Debug] ${message}:`, data);
  } else {
    console.log(`[Chat Debug] ${message}`);
  }
};

const store = writable<ChatState>(INITIAL_STATE);

const set = (val: Partial<ChatState>) => {
	store.update((state) => ({ ...state, ...val }));
};

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

const getActiveConversation = () => {
	const { conversations, activeConversationId } = get(store);
	if (!activeConversationId) {
		return null;
	}

	return conversations.find((c) => c.id === activeConversationId);
};

const insertMessageToActive = (message: Message) => {
	store.update((s) => {
		const conv = s.conversations.find((c) => c.id === s.activeConversationId);
		if (!conv) {
			return s;
		}
		conv.messages.push(message);
		return s;
	});
};

const removeMessageFromActive = (id: number) => {
	store.update((s) => {
		const conv = s.conversations.find((c) => c.id === s.activeConversationId);
		if (!conv) {
			return s;
		}
		conv.messages = conv.messages.filter((m) => m.id != id);
		return s;
	});
};

const scoreConversation = async (score: number) => {
	const conversationId = get(store).activeConversationId;

	return api.post(`/scores?conversation_id=${conversationId}`, { score });
};

/**
 * Extract recommended documents from the latest assistant message
 * to show document recommendations to the user
 */
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

const fetchConversations = async (documentId: number) => {
  set({ loading: true, error: '', lastMessageFailed: false });

  try {
    // Using path parameter instead of query parameter to match server route
    logDebug(`Fetching conversations for document: ${documentId}`);
    const { data } = await api.get(`/conversations/${documentId}`);
    logDebug("Conversation API response:", data);

    // Extract conversations from response object - server returns {conversations: [...]}
    const processedConversations = data.conversations || [];
    logDebug(`Received ${processedConversations.length} conversations`);

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
  } catch (error) {
    console.error("Error fetching conversations:", error);
    set({
      error: error.response?.data?.error || 'Failed to load conversations',
      loading: false,
      lastMessageFailed: true
    });
  }
}

const createConversation = async (documentId: number) => {
  set({ loading: true, error: '', lastMessageFailed: false });

  try {
    // Using path parameter instead of query parameter to match server route
    logDebug(`Creating conversation for document: ${documentId}`);
    const { data } = await api.post(`/conversations/${documentId}`);
    logDebug("Create conversation API response:", data);

    // Make sure we have a valid response before setting state
    if (data && data.id) {
      set({
        activeConversationId: data.id,
        conversations: [data, ...get(store).conversations],
        loading: false,
        // Reset research mode for new conversation
        researchMode: false,
        recommendedDocuments: [],
        activeDocuments: []
      });

      return data;
    } else {
      throw new Error("Invalid response data from server");
    }
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

const setActiveConversationId = (id: number) => {
	set({ activeConversationId: id, lastMessageFailed: false });

	// Update research mode state based on the newly selected conversation
	updateResearchModeFromActiveConversation();
};

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

    // CRITICAL FIX: Ensure we check both active flag AND multiple PDFs
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

// Add sendMessage function
const sendMessage = async (text: string, opts: MessageOpts = {}) => {
  const conversationId = get(store).activeConversationId;
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
  const currentState = get(store);
  const isResearchActive = currentState.researchMode;
  const activeDocuments = currentState.activeDocuments.map(doc => doc.id);

  logDebug(`Research mode: ${isResearchActive}, active docs:`, activeDocuments);

  // Add user message and pending message immediately for better UX
  const userMessage = {
    role: 'user',
    content: text
  };

  const pendingMessage = {
    role: 'pending',
    content: ''
  };

  store.update(s => {
    const conversation = s.conversations.find(c => c.id === conversationId);
    if (conversation) {
      conversation.messages = [...conversation.messages, userMessage, pendingMessage];
    }
    return s;
  });

  try {
    // FIX: Make sure we pass the correct parameters
    const response = await api.post(`/conversations/${conversationId}/messages`, {
      input: text,
      message: text, // Include for backward compatibility
      useStreaming: opts.useStreaming || false,
      useResearch: isResearchActive,
      activeDocs: activeDocuments
    });

    logDebug("Message response:", response.data);

    // Remove pending message
    store.update(s => {
      const conversation = s.conversations.find(c => c.id === conversationId);
      if (conversation) {
        conversation.messages = conversation.messages.filter(m => m.role !== 'pending');
      }
      return s;
    });

    // Process the response and update the conversation
    if (response.data) {
      // Extract the actual messages from the response
      const apiUserMessage = response.data.message || {
        role: 'user',
        content: text
      };

      const assistantMessage = response.data.response ? {
        role: 'assistant',
        content: response.data.response,
        metadata: response.data.metadata || {}
      } : response.data;

      // Add the messages to the conversation
      store.update(s => {
        const conversation = s.conversations.find(c => c.id === conversationId);
        if (conversation) {
          // Add user message if it's not included
          if (!conversation.messages.some(m => m.role === 'user' && m.content === text)) {
            conversation.messages.push(apiUserMessage);
          }

          // Add assistant message
          conversation.messages.push(assistantMessage);

          // Update research mode status if available
          if (response.data.research_mode || assistantMessage.metadata?.research_mode) {
            const researchMode = response.data.research_mode || assistantMessage.metadata?.research_mode;
            conversation.metadata = {
              ...conversation.metadata,
              research_mode: researchMode
            };

            logDebug("Updated research mode in conversation metadata:", researchMode);
          }
        }
        return s;
      });

      // Update application state based on the response
      updateResearchModeFromActiveConversation();

      set({ loading: false, lastMessageFailed: false });
      return response.data;
    } else {
      throw new Error("Invalid response from server");
    }
  } catch (error) {
    const errorMessage = error.response?.data?.error || error.message || 'Failed to send message';
    logDebug(`Error sending message: ${errorMessage}`, error);

    // Remove pending message on error
    store.update(s => {
      const conversation = s.conversations.find(c => c.id === conversationId);
      if (conversation) {
        conversation.messages = conversation.messages.filter(m => m.role !== 'pending');
      }
      return s;
    });

    set({
      error: errorMessage,
      loading: false,
      lastMessageFailed: true
    });

    return null;
  }
};

/**
 * Regenerate the response for a previously failed query with improved research mode handling
 * @param query The original query to regenerate a response for
 * @param options Options for message sending
 * @returns A Promise that resolves when the regeneration is complete
 */
const regenerateResponse = async (query: string, options: MessageOpts = {}) => {
  const currentState = get(store);
  const conversationId = currentState.activeConversationId;

  if (!conversationId) {
    console.error("No active conversation found");
    set({
      error: 'No active conversation. Please start a new chat or select an existing conversation.',
      loading: false,
      lastMessageFailed: true,
    });
    throw new Error('No active conversation');
  }

  set({ loading: true, error: '', lastMessageFailed: false });

  try {
    // Capture current research mode state before sending message
    const isResearchActive = currentState.researchMode;
    const activeDocuments = currentState.activeDocuments.map((doc) => doc.id);

    console.log(
      `Regenerating response with research mode: ${isResearchActive}, docs: ${activeDocuments.length}`
    );

    // Explicitly set research mode options
    options.useResearch = isResearchActive;
    options.activeDocs = activeDocuments;

    // Use the existing API endpoint with regeneration header
    const response = await api.post(
      `/conversations/${conversationId}/messages`,
      {
        input: query,
        useStreaming: options.useStreaming || false,
        useResearch: isResearchActive, // Use current state
        activeDocs: activeDocuments, // Send active doc IDs
      },
      { headers: { 'X-Regeneration-Request': 'true' } }
    );

    // Response should contain both user message and AI response
    const { message, response: assistantMessage } = response.data;

    if (!message || !assistantMessage) {
      throw new Error('Invalid response from server');
    }

    // Update the conversation - for regeneration, replace existing messages
    store.update((s) => {
      const conversation = s.conversations.find((c) => c.id === conversationId);
      if (!conversation) return s;

      // Remove pending or error messages
      conversation.messages = conversation.messages.filter(
        (m) =>
          m.role !== 'pending' &&
          !(m.role === 'assistant' && m.metadata?.status === 'error')
      );

      // Update conversation metadata if needed
      if (assistantMessage.metadata && assistantMessage.metadata.research_mode) {
        conversation.metadata = {
          ...conversation.metadata,
          research_mode: assistantMessage.metadata.research_mode,
        };

        console.log(
          "Updated research mode in metadata during regeneration:",
          assistantMessage.metadata.research_mode
        );
      }

      // Add the new response
      conversation.messages.push(assistantMessage);

      return s;
    });

    // Update research mode status based on assistant response
    if (assistantMessage.metadata) {
      // Extract recommended documents
      const recommendedDocs = assistantMessage.metadata.recommended_documents || [];

      // Extract active documents from research mode
      let updatedActiveDocs: DocumentInfo[] = [];
      let updatedResearchActive = false;

      if (assistantMessage.metadata.research_mode) {
        const researchMode = assistantMessage.metadata.research_mode;

        // Validate research mode properly
        updatedResearchActive =
          researchMode.active === true &&
          researchMode.pdf_ids &&
          researchMode.pdf_ids.length > 1;

        if (updatedResearchActive && researchMode.pdf_ids && researchMode.document_names) {
          updatedActiveDocs = researchMode.pdf_ids.map((id) => ({
            id,
            name: researchMode.document_names[id] || `Document ${id}`,
            isPrimary: id === String(message.conversation_id),
          }));
        }
      }

      set({
        recommendedDocuments: recommendedDocs,
        activeDocuments: updatedActiveDocs,
        researchMode: updatedResearchActive,
        loading: false,
        lastMessageFailed: false,
      });

      console.log(
        `Updated UI state after regeneration: research_mode=${updatedResearchActive}, active_docs=${updatedActiveDocs.length}`
      );
    } else {
      set({ loading: false, lastMessageFailed: false });
    }

    return { message, assistantMessage };
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
    const currentState = get(store);
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

    // CRITICAL FIX: Use the correct API endpoint
    // The server has a route at /conversations/<conversation_id>/research, NOT /research/activate
    const response = await api.post(`/conversations/${conversationId}/research`, {
      pdf_ids: formattedPdfIds,
      active: true  // Explicitly set active flag
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
    // FIX: Use the correct API endpoint for toggling research mode
    const response = await api.post(`/conversations/${conversationId}/research`, {
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

// Updated acceptRecommendedDocument function with improved research mode handling
const acceptRecommendedDocument = async (documentId: string) => {
  const state = get(store);
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

// New function to fetch available documents
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

// Add deleteDocument function since it's exported in index.ts
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

// Helper function to fetch document names
const fetchDocumentNames = async (pdfIds: string[]): Promise<DocumentInfo[]> => {
  try {
    // First, try to get names from existing documents in the store
    const existingDocs = get(store).availableDocuments;
    const cachedConversation = getActiveConversation();
    const primaryPdf = cachedConversation?.pdf;

    logDebug("Fetching document names for:", pdfIds);
    logDebug("Primary PDF:", primaryPdf);
    logDebug("Available documents cache:", existingDocs);

    // If we have cached documents, use those names first
    if (existingDocs.length > 0) {
      return pdfIds.map(id => {
        // Check if this is the primary document from conversation
        if (primaryPdf && String(primaryPdf.id) === String(id)) {
          return {
            id: id.toString(),
            name: primaryPdf.name || `Document ${id}`,
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
        return {
          id: id.toString(),
          name: data.name || data.title || `Document ${id}`,
          isPrimary: primaryPdf ? String(primaryPdf.id) === String(id) : false
        };
      } catch (error) {
        console.error(`Failed to fetch document ${id}:`, error);
        return {
          id: id.toString(),
          name: `Document ${id}`,
          isPrimary: primaryPdf ? String(primaryPdf.id) === String(id) : false
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

const resetAll = () => {
  set(INITIAL_STATE);
};

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
  // Research mode exports
  sendMessage,
  regenerateResponse,
  activateResearchMode,
  deactivateResearchMode,
  acceptRecommendedDocument,
  fetchAvailableDocuments,
  updateResearchModeFromActiveConversation,
  deleteDocument
};
