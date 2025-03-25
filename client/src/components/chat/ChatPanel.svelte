<script lang="ts">
  import { onMount, onDestroy, createEventDispatcher } from 'svelte';
  import { get } from 'svelte/store';
  import {
    store,
    resetError,
    fetchConversations,
    createConversation,
    getActiveConversation,
    activateResearchMode,
    deactivateResearchMode,
    acceptRecommendedDocument,
    fetchAvailableDocuments,
    sendMessage,
    regenerateResponse
  } from '$s/chat';
  import Alert from '$c/Alert.svelte';
  import ChatInput from '$c/chat/ChatInput.svelte';
  import ChatList from '$c/chat/ChatList.svelte';
  import ConversationSelect from '$c/chat/ConversationSelect.svelte';
  import ResearchModeIndicator from '$c/chat/ResearchModeIndicator.svelte';
  import DocumentSelectorModal from '$c/chat/DocumentSelectorModal.svelte';
  import RetryButton from '$c/chat/RetryButton.svelte';
  import Icon from '$c/Icon.svelte';

  export let documentId: number;

  // Use localStorage for feature toggles with safe default
  let useStreaming = typeof localStorage !== 'undefined' && !!localStorage.getItem('streaming');
  let isSubmitting = false; // Flag to prevent duplicate submissions
  let lastQuery = ""; // Store the last query for regeneration
  let isRegenerating = false; // Track regeneration state

  // State for document selection modal
  let showDocSelectorModal = false;
  let availableDocuments = [];
  let isLoadingDocuments = false;

  // Get research mode from store using Svelte's reactive syntax with safe defaults
  $: useResearch = $store?.researchMode || false;
  $: activeDocuments = $store?.activeDocuments || [];
  $: recommendedDocuments = $store?.recommendedDocuments || [];

  // Check for error messages to show regenerate button
  $: showRegenerateButton = !!$store?.error || ($store?.lastMessageFailed && lastQuery);

  const dispatch = createEventDispatcher();

  // Update localStorage based on changes (with safety checks)
  $: {
    if (typeof localStorage !== 'undefined') {
      localStorage.setItem('streaming', useStreaming ? 'true' : '');
    }
  }

  $: activeConversation = $store?.activeConversationId ? getActiveConversation() : null;

  async function handleSubmit(event: CustomEvent<string>) {
    try {
      // The raw message text (ensure it’s a string)
      const message = event?.detail || "";

      // Prevent empty or duplicate submissions
      if (!message || typeof message !== 'string' || !message.trim() || isSubmitting) {
        console.log("Message validation failed or already submitting", {
          message: `${message}`.substring(0, 30) + "...",
          isSubmitting
        });
        return;
      }

      lastQuery = message;       // Store the query for possible regeneration
      isSubmitting = true;       // Lock out duplicate submissions
      showRegenerateButton = false;

      // IMPORTANT: Pass the string `message` as the 1st arg,
      // and all additional fields in the 2nd arg:
      await sendMessage(message, {
        role: 'user',
        metadata: useResearch ? { research_mode: true } : undefined,
        useResearch,
        useStreaming
      });

    } catch (error) {
      console.error("Error sending message:", error);

      // Fallback approach: manual store update + direct fetch call
      try {
        const currentState = get(store);
        if (currentState?.activeConversationId) {
          // Show the user's message in the conversation immediately
          store.update(s => {
            const newState = JSON.parse(JSON.stringify(s));
            const conv = newState.conversations.find(
              c => c.id === newState.activeConversationId
            );

            if (conv) {
              if (!Array.isArray(conv.messages)) {
                conv.messages = [];
              }

              // Add the user's message
              conv.messages.push({
                role: 'user',
                content: lastQuery,
                id: Date.now()
              });

              // Add a pending placeholder
              conv.messages.push({
                role: 'pending',
                content: '',
                id: Date.now() + 1
              });
            }
            return newState;
          });

          // Make a direct POST to /api
          const result = await fetch(
            `/api/conversations/${currentState.activeConversationId}/messages`,
            {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json'
              },
              body: JSON.stringify({
                input: lastQuery,
                useResearch,
                useStreaming
              })
            }
          );

          if (result.ok) {
            // Refresh the conversation on success
            fetchConversations(documentId);
          } else {
            // Show error in the store
            const errorData = await result.json();
            console.error("API error:", errorData);
            store.update(s => ({ ...s, error: errorData.error || "Failed to send message" }));
          }
        }
      } catch (fallbackError) {
        console.error("Fallback error:", fallbackError);
      }
    } finally {
      isSubmitting = false; // Unlock submission
    }
  }

  // New function to handle regeneration
  async function handleRegenerate() {
    if (!lastQuery || typeof lastQuery !== 'string' || isRegenerating) {
      return; // Don't proceed if no valid query or already regenerating
    }

    isRegenerating = true;

    // Add a pending message to show loading state
    store.update(s => {
      // Safety check for valid state
      if (!s || !s.conversations) return s;

      const conv = s.conversations.find(c => c.id === s.activeConversationId);
      if (conv) {
        // Add a pending message to show loading
        if (!Array.isArray(conv.messages)) {
          conv.messages = [];
        }

        conv.messages.push({
          role: 'pending',
          content: '',
          id: Date.now()
        });
      }
      return s;
    });

    try {
      // Use the regenerateResponse function from the store
      await regenerateResponse(lastQuery, { useStreaming, useResearch });
    } catch (error) {
      console.error("Error regenerating response:", error);
    } finally {
      isRegenerating = false;

      // Remove any pending messages that might still be there
      store.update(s => {
        // Safety check for valid state
        if (!s || !s.conversations) return s;

        const conv = s.conversations.find(c => c.id === s.activeConversationId);
        if (conv && Array.isArray(conv.messages)) {
          conv.messages = conv.messages.filter(m => m.role !== 'pending');
        }
        return s;
      });
    }
  }

  function handleNewChat() {
    createConversation(documentId);
    lastQuery = ""; // Reset last query for new conversation
    showRegenerateButton = false; // Hide regenerate button for new conversation
  }

  function handleResearchToggle() {
    if (!$store?.researchMode && !showDocSelectorModal) {
      // When enabling research mode, show document selector
      showDocumentSelector();
    } else if ($store?.researchMode) {
      // When disabling, deactivate research mode
      try {
        // Use proper export from index.ts
        import('$s/chat/index').then(module => {
          const deactivateResearchMode = module.deactivateResearchMode;
          if (deactivateResearchMode && $store?.activeConversationId) {
            deactivateResearchMode($store.activeConversationId);
          } else {
            // Fallback UI update
            store.update(s => ({
              ...s,
              researchMode: false,
              activeDocuments: []
            }));
          }
        }).catch(err => {
          console.error("Error importing deactivateResearchMode:", err);
          // Fallback UI update
          store.update(s => ({
            ...s,
            researchMode: false,
            activeDocuments: []
          }));
        });
      } catch (error) {
        console.error("Error deactivating research mode:", error);
      }
    }
  }

  function showDocumentSelector() {
    isLoadingDocuments = true;

    try {
      // Fetch available documents for research
      fetchAvailableDocuments().then(docs => {
        // Get the primary document ID (the current document)
        const primaryDocId = documentId.toString();

        // Filter out the primary document from the available list
        availableDocuments = docs.filter(doc => doc.id.toString() !== primaryDocId);

        console.log("Primary document filtered out:", primaryDocId);
        console.log("Available documents for selection:", availableDocuments);

        showDocSelectorModal = true;
        isLoadingDocuments = false;
      }).catch(error => {
        console.error("Failed to fetch available documents:", error);
        isLoadingDocuments = false;
      });
    } catch (error) {
      console.error("Failed to show document selector:", error);
      isLoadingDocuments = false;
    }
  }

  async function handleActivateResearch(event) {
    const { conversationId, documentIds } = event.detail || {};

    if (!conversationId || !documentIds) {
      console.error("Missing required parameters for research mode activation");
      return;
    }

    try {
      // Format the request properly
      const formattedDocumentIds = Array.isArray(documentIds)
        ? documentIds.map(id => id.toString()).filter(id => id && id.trim().length > 0)
        : [];

      if (formattedDocumentIds.length < 2) {
        throw new Error("At least two valid document IDs are required for research mode");
      }

      // Make sure to include the primary document (current document)
      if (!formattedDocumentIds.includes(documentId.toString())) {
        formattedDocumentIds.unshift(documentId.toString());
      }

      console.log("Activating research mode with documents:", formattedDocumentIds);

      // Call activateResearchMode with the properly formatted document IDs
      await activateResearchMode(conversationId, formattedDocumentIds);

      // Ensure the primary document info is properly passed to the active documents
      store.update(s => {
        // Skip if store is invalid
        if (!s || !Array.isArray(s.activeDocuments)) return s;

        // If the primary document wasn't already in activeDocuments, make sure to add it
        const primaryExists = s.activeDocuments.some(doc =>
          doc.id.toString() === documentId.toString()
        );

        if (!primaryExists && s.activeDocuments.length > 0) {
          // Get current document info from conversation
          const activeConversation = getActiveConversation();
          if (activeConversation && activeConversation.pdf) {
            // Add primary document to active documents with proper info
            s.activeDocuments.unshift({
              id: documentId.toString(),
              name: activeConversation.pdf.name || `Document ${documentId}`,
              isPrimary: true
            });
          }
        }

        return s;
      });

    } catch (error) {
      console.error("Failed to activate research mode:", error);
      // Show friendly error to user
      store.update(s => ({
        ...s,
        error: "Failed to activate research mode. Please try again."
      }));
    }
  }

  async function handleAddDocument(event) {
    try {
      const documentId = event?.detail?.documentId;
      if (!documentId) {
        console.error("No document ID provided for adding");
        return;
      }

      await acceptRecommendedDocument(documentId);
    } catch (error) {
      console.error("Failed to add recommended document:", error);
    }
  }

  // Consolidated handleRemoveDocument function
  async function handleRemoveDocument(event) {
    try {
      const documentId = event?.detail?.documentId;
      if (!documentId) {
        console.error("No document ID provided for removal");
        return;
      }

      console.log("Handling document removal:", documentId);

      // Get current active documents
      const activeConversation = getActiveConversation();
      if (!activeConversation) {
        console.error("No active conversation");
        return;
      }

      // Get current PDF IDs from the active documents excluding the one to remove
      const currentDocIds = $store?.activeDocuments
        ?.filter(doc => doc && doc.id !== documentId)
        ?.map(doc => doc.id) || [];

      // Reactivate research mode with the filtered list of documents
      if (currentDocIds.length > 0) {
        await activateResearchMode($store?.activeConversationId, currentDocIds);
      } else {
        // If no documents left, deactivate research mode
        await deactivateResearchMode($store?.activeConversationId);
      }
    } catch (error) {
      console.error("Failed to remove document:", error);
      // Show friendly error to user
      store.update(s => ({
        ...s,
        error: "Failed to remove document. Please try again."
      }));
    }
  }

  function handleOpenDocument(event) {
    console.log("Opening document from ChatPanel:", event.detail);
    dispatch('openDocument', event.detail);
  }

  function handleLinkClick(event: MouseEvent) {
    const target = event.target as HTMLAnchorElement;
    if (target && target.tagName === 'A' && target.href && target.href.startsWith('pdf-search://')) {
      event.preventDefault();
      const [, pdfId, pageNumber, elementId] = target.href.split('/');
      dispatch('pdfSearch', { pdfId, pageNumber: parseInt(pageNumber), elementId });
    }
  }

  function cleanupActiveDocuments() {
    // Get current research mode state
    const isResearchActive = $store?.researchMode || false;
    const activeConversationId = $store?.activeConversationId;

    if (isResearchActive && activeConversationId) {
      // Get all valid document IDs from active documents
      const validDocumentIds = ($store?.activeDocuments || [])
        .filter(doc =>
          // Document must have a valid ID and name
          doc && doc.id &&
          doc.name &&
          // Filter out generic "Document" entries with no proper name
          (doc.name !== "Document" && doc.name !== `Document ${doc.id}`)
        )
        .map(doc => doc.id);

      console.log("Cleaning up active documents, valid IDs:", validDocumentIds);

      // If we have at least one valid document, reactivate research mode with only valid documents
      if (validDocumentIds.length > 0) {
        // Reactivate research mode with only valid documents
        activateResearchMode(activeConversationId, validDocumentIds)
          .then(() => console.log("Successfully cleaned up active documents"))
          .catch(err => console.error("Error cleaning up active documents:", err));
      } else if (isResearchActive) {
        // If no valid documents but research mode is active, deactivate it
        deactivateResearchMode(activeConversationId)
          .then(() => console.log("Deactivated research mode due to no valid documents"))
          .catch(err => console.error("Error deactivating research mode:", err));
      }
    }
  }

  // Add to onMount in ChatPanel.svelte
  onMount(() => {
    fetchConversations(documentId);
    document.addEventListener('click', handleLinkClick);

    // Add a slight delay to ensure conversation is loaded
    setTimeout(cleanupActiveDocuments, 500);
  });

  onDestroy(() => {
    document.removeEventListener('click', handleLinkClick);
  });
</script>

<div class="chat-panel">
  <!-- Header with controls -->
  <div class="chat-header">
    <div class="flex items-center space-x-2">
      <div class="chat-icon">
        <Icon name="chat" size="18px" />
      </div>
      <h2 class="header-title">Document Chat</h2>
    </div>

    <div class="flex items-center space-x-2">
      <!-- Toggle controls -->
      <div class="toggle-group">
        <button
          class="toggle-button {useStreaming ? 'active' : ''}"
          on:click={() => useStreaming = true}
          title="Enable streaming responses"
        >
          <Icon name="autorenew" size="14px" class="mr-1" />
          Streaming
        </button>
        <button
          class="toggle-button {!useStreaming ? 'active' : ''}"
          on:click={() => useStreaming = false}
          title="Disable streaming responses"
        >
          <Icon name="done_all" size="14px" class="mr-1" />
          Standard
        </button>
      </div>

      <!-- Research mode toggle -->
      <button
        class="research-toggle {useResearch ? 'active' : ''}"
        on:click={handleResearchToggle}
        title={useResearch ? "Disable research mode" : "Enable research mode"}
      >
        <Icon name="search" size="14px" class="mr-1" />
        Research {useResearch ? 'On' : 'Off'}
      </button>

      {#if useResearch}
        <button
          class="icon-button"
          on:click={showDocumentSelector}
          title="Select Documents"
        >
          <Icon name="folder_open" size="18px" />
        </button>
      {/if}
    </div>
  </div>

  <!-- Conversation selector -->
  <div class="conversation-controls">
    <ConversationSelect conversations={$store?.conversations || []} />
    <button
      class="new-chat-button"
      on:click={handleNewChat}
    >
      <Icon name="add" size="16px" class="mr-1" />
      New Chat
    </button>
  </div>

  <!-- Main chat area -->
  <div class="chat-content">
    {#if useResearch && activeDocuments.length > 0}
      <div class="px-4 py-2">
        <ResearchModeIndicator
          activeDocuments={activeDocuments}
          recommendedDocuments={recommendedDocuments}
          primaryDocumentId={documentId.toString()}
          on:addDocument={handleAddDocument}
          on:removeDocument={handleRemoveDocument}
          on:showDocumentSelector={showDocumentSelector}
          on:openDocument={handleOpenDocument}
        />
      </div>
    {/if}

    <!-- Messages list -->
    <div class="messages-container">
      <ChatList messages={activeConversation?.messages || []} />
    </div>

    <!-- Input area -->
    <div class="input-container">
      {#if $store?.error}
        <div class="mb-3">
          <Alert type="error" message={$store.error} onClose={resetError} />
        </div>
      {/if}

      {#if showRegenerateButton && lastQuery}
        <div class="regenerate-container">
          <RetryButton
            on:click={handleRegenerate}
            isLoading={isRegenerating}
          />
        </div>
      {/if}

      <ChatInput
        on:submit={handleSubmit}
        disabled={$store?.loading || isSubmitting || isRegenerating}
        loading={$store?.loading || isSubmitting || isRegenerating}
      />
    </div>
  </div>

  <!-- Document selector modal -->
  <DocumentSelectorModal
    isOpen={showDocSelectorModal}
    conversation={activeConversation}
    availableDocuments={availableDocuments}
    on:close={() => showDocSelectorModal = false}
    on:activateResearch={handleActivateResearch}
  />
</div>

<style>
  /* Styles remain the same */
  .chat-panel {
    display: flex;
    flex-direction: column;
    height: 100%;
    background-color: white;
    overflow: hidden;
    --chat-primary: #3b82f6;
    --chat-primary-dark: #2563eb;
    --chat-header-bg: white;
    --chat-content-bg: #f8fafc;
  }

  .chat-panel {
    display: flex;
    flex-direction: column;
    height: 100%;
    background-color: white;
    overflow: hidden;
    --chat-primary: #3b82f6;
    --chat-primary-dark: #2563eb;
    --chat-header-bg: white;
    --chat-content-bg: #f8fafc;
  }

  .chat-header {
    padding: 12px 16px;
    border-bottom: 1px solid #e5e7eb;
    background-color: var(--chat-header-bg);
    display: flex;
    justify-content: space-between;
    align-items: center;
  }

  .chat-icon {
    width: 28px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 6px;
    background-color: #ebf5ff;
    color: var(--chat-primary);
  }

  .header-title {
    font-size: 14px;
    font-weight: 600;
    color: #1f2937;
  }

  .toggle-group {
    display: flex;
    background-color: #f1f5f9;
    border-radius: 6px;
    padding: 2px;
  }

  .toggle-button {
    font-size: 12px;
    padding: 4px 8px;
    border-radius: 4px;
    border: none;
    background: transparent;
    color: #64748b;
    display: flex;
    align-items: center;
    transition: all 0.2s ease;
  }

  .toggle-button.active {
    background-color: var(--chat-primary);
    color: white;
  }

  .research-toggle {
    font-size: 12px;
    padding: 4px 8px;
    border-radius: 4px;
    border: none;
    background-color: #f1f5f9;
    color: #64748b;
    display: flex;
    align-items: center;
    transition: all 0.2s ease;
  }

  .research-toggle.active {
    background-color: #ebf5ff;
    color: var(--chat-primary);
  }

  .icon-button {
    width: 28px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 6px;
    border: none;
    background-color: #f1f5f9;
    color: #64748b;
    transition: all 0.2s ease;
  }

  .icon-button:hover, .research-toggle:hover {
    background-color: #e2e8f0;
  }

  .conversation-controls {
    padding: 10px 16px;
    background-color: #f8fafc;
    border-bottom: 1px solid #e5e7eb;
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .new-chat-button {
    display: flex;
    align-items: center;
    padding: 6px 10px;
    font-size: 12px;
    font-weight: 500;
    border-radius: 6px;
    background-color: var(--chat-primary);
    color: white;
    border: none;
    transition: all 0.2s ease;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
  }

  .new-chat-button:hover {
    background-color: var(--chat-primary-dark);
  }

  .chat-content {
    flex: 1;
    display: flex;
    flex-direction: column;
    overflow: hidden;
  }

  .messages-container {
    flex: 1;
    overflow: auto;
    padding: 16px;
    background-color: var(--chat-content-bg);
  }

  .input-container {
    padding: 12px 16px;
    border-top: 1px solid #e5e7eb;
    background-color: white;
  }

  .regenerate-container {
    display: flex;
    justify-content: center;
    margin-bottom: 12px;
  }

  /* Smooth scrollbar for messages */
  .messages-container {
    scrollbar-width: thin;
    scrollbar-color: #cbd5e1 transparent;
  }

  .messages-container::-webkit-scrollbar {
    width: 6px;
  }

  .messages-container::-webkit-scrollbar-track {
    background: transparent;
  }

  .messages-container::-webkit-scrollbar-thumb {
    background-color: #cbd5e1;
    border-radius: 3px;
  }
</style>
