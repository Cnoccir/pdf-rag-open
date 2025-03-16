<!-- DocumentSidebar.svelte -->
<script lang="ts">
  import { createEventDispatcher, onMount } from 'svelte';
  import Icon from '$c/Icon.svelte';
  import { store } from '$s/chat'; // Import store to access document info if needed

  export let pdfId: string;
  export let isOutlineVisible: boolean = true;
  export let documentName: string = "Document";

  let activeTab = 'outline'; // outline, search, bookmarks, history
  let searchQuery = '';
  let searchResults = [];
  let isSearching = false;
  let document = null;

  const dispatch = createEventDispatcher();

  function toggleOutline() {
    dispatch('toggleOutline');
  }

  function setActiveTab(tab: string) {
    activeTab = tab;
    dispatch('tabChange', { tab });
  }

  function performSearch() {
    if (!searchQuery.trim()) return;

    isSearching = true;
    // Here we would normally call a search function
    // For now, we'll just simulate it
    setTimeout(() => {
      searchResults = [
        { page: 3, text: "Found text that matches your search query", snippet: "...context around the found text..." },
        { page: 7, text: "Another match for your search", snippet: "...more context..." },
      ];
      isSearching = false;
    }, 500);

    dispatch('search', { query: searchQuery });
  }

  function handleSearchKeydown(e) {
    if (e.key === 'Enter') performSearch();
  }

  function gotoSearchResult(page) {
    dispatch('gotoPage', { page });
  }

  onMount(async () => {
    // Fetch document details if needed
    try {
      // This would normally be an API call
      // const response = await fetch(`/api/documents/${pdfId}`);
      // document = await response.json();
      // documentName = document.name || "Document";
    } catch (error) {
      console.error('Error loading document details:', error);
    }
  });
</script>

<div class="sidebar">
  <div class="sidebar-header">
    <h2 class="document-title">
      <Icon name="description" size="18px" class="title-icon" />
      {documentName}
    </h2>
    <button
      class="toggle-button"
      on:click={toggleOutline}
      aria-label="Toggle sidebar"
    >
      <Icon name="chevron_left" size="20px" />
    </button>
  </div>

  <div class="sidebar-tabs">
    <button
      class="tab-button {activeTab === 'outline' ? 'active' : ''}"
      on:click={() => setActiveTab('outline')}
      title="Document Outline"
    >
      <Icon name="menu_book" size="18px" />
      <span class="tab-label">Outline</span>
    </button>
    <button
      class="tab-button {activeTab === 'search' ? 'active' : ''}"
      on:click={() => setActiveTab('search')}
      title="Search Document"
    >
      <Icon name="search" size="18px" />
      <span class="tab-label">Search</span>
    </button>
    <button
      class="tab-button {activeTab === 'bookmarks' ? 'active' : ''}"
      on:click={() => setActiveTab('bookmarks')}
      title="Bookmarks"
    >
      <Icon name="bookmark" size="18px" />
      <span class="tab-label">Bookmarks</span>
    </button>
    <button
      class="tab-button {activeTab === 'history' ? 'active' : ''}"
      on:click={() => setActiveTab('history')}
      title="History"
    >
      <Icon name="history" size="18px" />
      <span class="tab-label">History</span>
    </button>
  </div>

  <div class="sidebar-content">
    {#if activeTab === 'outline'}
      <div class="content-section">
        <h3 class="section-title">
          <Icon name="format_list_bulleted" size="16px" class="section-icon" />
          Document Outline
        </h3>
        <div class="empty-state">
          <Icon name="menu_book" size="24px" class="empty-icon" />
          <p>Document outline is not available for this PDF.</p>
          <button class="action-button">Generate Outline</button>
        </div>
      </div>
    {:else if activeTab === 'search'}
      <div class="content-section">
        <h3 class="section-title">
          <Icon name="search" size="16px" class="section-icon" />
          Search Document
        </h3>
        <div class="search-input-container">
          <input
            type="text"
            placeholder="Search document..."
            class="search-input"
            bind:value={searchQuery}
            on:keydown={handleSearchKeydown}
          />
          <button
            class="search-button"
            on:click={performSearch}
            title="Search"
          >
            <Icon name="search" size="18px" />
          </button>
        </div>

        {#if isSearching}
          <div class="search-loading">
            <div class="loading-spinner"></div>
            <span>Searching...</span>
          </div>
        {:else if searchResults.length > 0}
          <div class="search-results">
            <h4 class="results-summary">
              <Icon name="analytics" size="14px" class="summary-icon" />
              {searchResults.length} results found
            </h4>
            {#each searchResults as result}
              <div
                class="result-item"
                on:click={() => gotoSearchResult(result.page)}
                on:keydown={(e) => e.key === 'Enter' && gotoSearchResult(result.page)}
                tabindex="0"
                role="button"
              >
                <div class="result-page">
                  <Icon name="description" size="14px" class="page-icon" />
                  Page {result.page}
                </div>
                <div class="result-text">{result.text}</div>
                <div class="result-snippet">{result.snippet}</div>
              </div>
            {/each}
          </div>
        {:else if searchQuery}
          <div class="empty-state">
            <Icon name="search_off" size="24px" class="empty-icon" />
            <p>No results found</p>
          </div>
        {/if}
      </div>
    {:else if activeTab === 'bookmarks'}
      <div class="content-section">
        <h3 class="section-title">
          <Icon name="bookmark" size="16px" class="section-icon" />
          Bookmarks
        </h3>
        <div class="empty-state">
          <Icon name="bookmark_border" size="24px" class="empty-icon" />
          <p>No bookmarks added yet</p>
          <button class="action-button">
            <Icon name="add" size="14px" class="action-icon" />
            Add bookmark
          </button>
        </div>
      </div>
    {:else if activeTab === 'history'}
      <div class="content-section">
        <h3 class="section-title">
          <Icon name="history" size="16px" class="section-icon" />
          Recent Documents
        </h3>
        <div class="empty-state">
          <Icon name="folder_open" size="24px" class="empty-icon" />
          <p>No recent documents</p>
        </div>
      </div>
    {/if}
  </div>
</div>

<style>
  .sidebar {
    display: flex;
    flex-direction: column;
    height: 100%;
    background-color: white;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    --sidebar-primary: #3b82f6;
    --sidebar-text: #1f2937;
    --sidebar-bg: white;
    --sidebar-border: #e5e7eb;
  }

  .sidebar-header {
    padding: 16px;
    border-bottom: 1px solid var(--sidebar-border);
    display: flex;
    align-items: center;
    justify-content: space-between;
  }

  .document-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 14px;
    font-weight: 600;
    color: var(--sidebar-text);
    margin: 0;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
  }

  .title-icon {
    color: var(--sidebar-primary);
    flex-shrink: 0;
  }

  .toggle-button {
    width: 28px;
    height: 28px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 50%;
    border: none;
    background-color: transparent;
    color: #64748b;
    cursor: pointer;
    transition: background-color 0.2s;
  }

  .toggle-button:hover {
    background-color: #f1f5f9;
  }

  .sidebar-tabs {
    display: flex;
    gap: 4px;
    padding: 8px;
    border-bottom: 1px solid var(--sidebar-border);
    background-color: #f8fafc;
  }

  .tab-button {
    flex: 1;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 8px 4px;
    border-radius: 6px;
    border: none;
    background-color: transparent;
    color: #64748b;
    cursor: pointer;
    transition: all 0.2s;
  }

  .tab-button:hover {
    background-color: #f1f5f9;
  }

  .tab-button.active {
    background-color: #e0f2fe;
    color: var(--sidebar-primary);
  }

  .tab-label {
    font-size: 11px;
    margin-top: 4px;
  }

  .sidebar-content {
    flex: 1;
    overflow-y: auto;
    padding: 16px;
  }

  .content-section {
    display: flex;
    flex-direction: column;
    gap: 12px;
  }

  .section-title {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 14px;
    font-weight: 600;
    color: var(--sidebar-text);
    margin: 0;
  }

  .section-icon {
    color: var(--sidebar-primary);
  }

  .empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 24px 16px;
    background-color: #f8fafc;
    border-radius: 8px;
    border: 1px solid #e2e8f0;
    text-align: center;
  }

  .empty-icon {
    color: #94a3b8;
  }

  .empty-state p {
    font-size: 13px;
    color: #64748b;
    margin: 0;
  }

  .action-button {
    display: flex;
    align-items: center;
    gap: 4px;
    padding: 6px 12px;
    border-radius: 4px;
    border: none;
    background-color: #f1f5f9;
    color: var(--sidebar-primary);
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
    transition: background-color 0.2s;
    margin-top: 8px;
  }

  .action-button:hover {
    background-color: #e0f2fe;
  }

  .search-input-container {
    position: relative;
  }

  .search-input {
    width: 100%;
    padding: 8px 36px 8px 12px;
    border-radius: 6px;
    border: 1px solid #e2e8f0;
    font-size: 13px;
    outline: none;
    transition: all 0.2s;
  }

  .search-input:focus {
    border-color: var(--sidebar-primary);
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1);
  }

  .search-button {
    position: absolute;
    right: 8px;
    top: 50%;
    transform: translateY(-50%);
    background: none;
    border: none;
    color: #64748b;
    cursor: pointer;
    padding: 4px;
    border-radius: 4px;
    transition: all 0.2s;
  }

  .search-button:hover {
    background-color: #f1f5f9;
    color: var(--sidebar-primary);
  }

  .search-loading {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 8px;
    padding: 16px;
    font-size: 13px;
    color: #64748b;
  }

  .loading-spinner {
    width: 16px;
    height: 16px;
    border: 2px solid #e2e8f0;
    border-top-color: var(--sidebar-primary);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .search-results {
    display: flex;
    flex-direction: column;
    gap: 8px;
  }

  .results-summary {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 12px;
    font-weight: 500;
    color: #64748b;
    margin: 4px 0;
  }

  .summary-icon {
    color: var(--sidebar-primary);
  }

  .result-item {
    padding: 10px;
    border-radius: 6px;
    background-color: white;
    border: 1px solid #e2e8f0;
    cursor: pointer;
    transition: all 0.2s;
  }

  .result-item:hover {
    background-color: #f8fafc;
    border-color: #cbd5e1;
  }

  .result-page {
    display: flex;
    align-items: center;
    gap: 4px;
    font-size: 12px;
    font-weight: 600;
    color: var(--sidebar-primary);
    margin-bottom: 4px;
  }

  .page-icon {
    flex-shrink: 0;
  }

  .result-text {
    font-size: 13px;
    font-weight: 500;
    color: var(--sidebar-text);
    margin-bottom: 4px;
  }

  .result-snippet {
    font-size: 12px;
    color: #64748b;
    word-break: break-word;
  }

  /* Custom scrollbar */
  .sidebar-content {
    scrollbar-width: thin;
    scrollbar-color: #cbd5e1 transparent;
  }

  .sidebar-content::-webkit-scrollbar {
    width: 4px;
  }

  .sidebar-content::-webkit-scrollbar-track {
    background: transparent;
  }

  .sidebar-content::-webkit-scrollbar-thumb {
    background-color: #cbd5e1;
    border-radius: 2px;
  }
</style>
