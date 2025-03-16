<!-- ExpandableChatPanel.svelte -->
<script lang="ts">
  import { onMount, createEventDispatcher } from 'svelte';
  import ChatPanel from './ChatPanel.svelte';
  import Icon from '$c/Icon.svelte';

  export let onSubmit: (text: string, useStreaming: boolean, useResearch: boolean) => void;
  export let documentId: number;
  export let pdfId: string;
  export let onPdfSearch: (pageNumber: number, elementId: string) => void;

  let isExpanded = false;
  let isMobileView = false;
  let chatPanelWidth = '33%';

  const dispatch = createEventDispatcher();

  function toggleExpand() {
    isExpanded = !isExpanded;
    chatPanelWidth = isExpanded ? '50%' : '33%';
    dispatch('resize', chatPanelWidth);
  }

  function handlePdfSearch(event: CustomEvent) {
    const { pageNumber, elementId } = event.detail;
    onPdfSearch(pageNumber, elementId);
  }

  function handleOpenDocument(event) {
    // ENHANCED: Add detailed logging
    console.log("ExpandableChatPanel: Received openDocument event", event.detail);

    // CRITICAL: Make sure to dispatch the event with the correct detail
    dispatch('openDocument', event.detail);

    console.log("ExpandableChatPanel: Dispatched openDocument event");
  }

  onMount(() => {
    dispatch('resize', chatPanelWidth);

    const mediaQuery = window.matchMedia('(max-width: 768px)');
    isMobileView = mediaQuery.matches;

    const handleResize = () => {
      isMobileView = mediaQuery.matches;
    };

    mediaQuery.addEventListener('change', handleResize);

    return () => {
      mediaQuery.removeEventListener('change', handleResize);
    };
  });
</script>

<div class="expandable-chat-panel {$$props.class || ''} {isExpanded ? 'expanded' : ''}"
     style={!isMobileView ? `width: ${chatPanelWidth}` : ''}>
  <div class="resize-indicator">
    <div class="resize-handle"></div>
  </div>

  <div class="chat-panel-header bg-white">
    {#if isMobileView}
      <div
        class="mobile-chat-handle"
        on:click={toggleExpand}
        on:keydown={(e) => e.key === 'Enter' && toggleExpand()}
        tabindex="0"
        role="button"
        aria-label="Toggle chat panel"
      >
        <div class="chat-handle-indicator"></div>
        <div class="chat-handle-icon">
          <span class="text-sm font-medium text-gray-600">Document Chat</span>
          <Icon name={isExpanded ? "expand_more" : "expand_less"} size="18px" class="text-gray-600" />
        </div>
      </div>
    {:else}
      <button
        class="expand-toggle-btn"
        on:click={toggleExpand}
        title={isExpanded ? "Collapse chat" : "Expand chat"}
      >
        <Icon name={isExpanded ? "chevron_right" : "chevron_left"} size="18px" />
        <span class="ml-1 text-sm font-medium">{isExpanded ? "Collapse" : "Expand"}</span>
      </button>
    {/if}
  </div>

  <ChatPanel
    {onSubmit}
    {documentId}
    {pdfId}
    on:pdfSearch={handlePdfSearch}
    on:openDocument={handleOpenDocument}
  />
</div>

<style>
  .expandable-chat-panel {
    height: 100%;
    background-color: white;
    transition: width 0.3s ease, transform 0.3s ease;
    box-shadow: -5px 0 20px rgba(0, 0, 0, 0.08);
    display: flex;
    flex-direction: column;
    position: relative;
    z-index: 30;
    border-left: 1px solid #e5e7eb;
  }

  .chat-panel-header {
    padding: 10px 0;
    border-bottom: 1px solid #f1f5f9;
    z-index: 31;
    display: flex;
    align-items: center;
  }

  .expand-toggle-btn {
    display: flex;
    align-items: center;
    padding: 6px 10px;
    margin-left: 10px;
    background-color: #f8fafc;
    border-radius: 6px;
    color: #4b5563;
    border: none;
    cursor: pointer;
    transition: all 0.2s ease;
    box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
  }

  .expand-toggle-btn:hover {
    background-color: #f1f5f9;
    color: #1e40af;
  }

  .mobile-chat-handle {
    width: 100%;
    height: 40px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    border-top-left-radius: 16px;
    border-top-right-radius: 16px;
    background: white;
    box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.05);
  }

  .chat-handle-indicator {
    width: 36px;
    height: 4px;
    background-color: #e2e8f0;
    border-radius: 2px;
    margin-bottom: 8px;
  }

  .chat-handle-icon {
    display: flex;
    align-items: center;
    gap: 4px;
  }

  .resize-indicator {
    position: absolute;
    left: -3px;
    top: 0;
    bottom: 0;
    width: 6px;
    z-index: 35;
    cursor: col-resize;
    display: flex;
    align-items: center;
  }

  .resize-handle {
    width: 3px;
    height: 40px;
    background-color: #e2e8f0;
    border-radius: 1.5px;
    transition: background-color 0.2s;
    opacity: 0;
    transition: opacity 0.2s;
  }

  .resize-indicator:hover .resize-handle {
    opacity: 1;
    background-color: #3b82f6;
  }

  /* Mobile view styles */
  @media (max-width: 768px) {
    .expandable-chat-panel {
      position: fixed;
      bottom: 0;
      left: 0;
      right: 0;
      height: 60%;
      width: 100% !important;
      box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.15);
      border-radius: 16px 16px 0 0;
      transform: translateY(calc(100% - 40px));
      border-left: none;
    }

    .expandable-chat-panel.expanded {
      transform: translateY(0);
    }

    .resize-indicator {
      display: none;
    }
  }
</style>
