<script lang="ts">
  import { onMount } from 'svelte';
  import { page } from '$app/stores';
  import { goto } from '$app/navigation';
  import type { PageData } from './$types';
  import { triggerEmbeddingProcess, softDeleteDocument } from './apiHelpers';
  import DocumentSummary from '$c/DocumentSummary.svelte';
  import * as pdfjs from 'pdfjs-dist';

  // Initialize PDF.js worker
  pdfjs.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`;

  export let data: PageData;

  let documents = data.documents || [];
  let loading = true;
  let error: string | null = null;
  let searchQuery = '';
  let selectedCategory = 'all';
  let viewMode = 'grid';
  let sortBy = 'name';
  let showSummaries = true; // Toggle for showing summaries
  let loadingDetails = false; // Track detailed metadata loading

  const categories = [
    { id: 'all', name: 'All Documents' },
    { id: 'honeywell', name: 'Honeywell' },
    { id: 'tridium', name: 'Tridium' },
    { id: 'johnson', name: 'Johnson Controls' },
    { id: 'general', name: 'General Knowledge' }
  ];

  $: filteredDocuments = documents
    .filter(doc => {
      const matchesSearch = searchQuery === '' ||
        doc.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
        (doc.description || '').toLowerCase().includes(searchQuery.toLowerCase());
      const matchesCategory = selectedCategory === 'all' || doc.category === selectedCategory;
      return matchesSearch && matchesCategory;
    })
    .sort((a, b) => {
      switch (sortBy) {
        case 'name':
          return a.name.localeCompare(b.name);
        case 'date':
          return new Date(b.updatedAt || 0) - new Date(a.updatedAt || 0);
        default:
          return 0;
      }
    });

  async function generateThumbnail(doc) {
    try {
      if (!doc.thumbnail) {
        const loadingTask = pdfjs.getDocument(`/api/pdfs/${doc.id}/content`);
        const pdf = await loadingTask.promise;
        const page = await pdf.getPage(1);
        const viewport = page.getViewport({ scale: 0.5 });
        const canvas = document.createElement('canvas');
        const context = canvas.getContext('2d');

        canvas.width = viewport.width;
        canvas.height = viewport.height;

        await page.render({
          canvasContext: context,
          viewport: viewport
        }).promise;

        doc.thumbnail = canvas.toDataURL();
      }
    } catch (error) {
      console.error('Error generating thumbnail:', error);
    }
  }

  // Enhanced function to load detailed document metadata
  async function loadDocumentDetails() {
    if (!documents.length) return;

    loadingDetails = true;

    try {
      // Process 3 documents at a time to avoid overwhelming the server
      for (let i = 0; i < documents.length; i += 3) {
        const batch = documents.slice(i, i + 3);

        // Fetch detailed info for each document in the batch
        await Promise.all(batch.map(async (doc) => {
          try {
            // Fetch detailed document info (includes summaries)
            const response = await fetch(`/api/pdfs/${doc.id}`);
            if (response.ok) {
              const data = await response.json();

              // Update document with data from API
              if (data.pdf) {
                Object.assign(doc, data.pdf);
              }

              // Initialize metadata if it doesn't exist
              if (!doc.metadata) {
                doc.metadata = {};
              }

              // Handle summary data
              if (data.summary) {
                // Store the summary in document_summary field
                doc.metadata.document_summary = data.summary;
                console.log(`Loaded summary for ${doc.name}:`, data.summary);
              } else if (data.pdf?.metadata?.document_summary) {
                // Alternative: check if summary is in pdf.metadata
                doc.metadata.document_summary = data.pdf.metadata.document_summary;
                console.log(`Loaded summary from metadata for ${doc.name}:`, doc.metadata.document_summary);
              }

              // Store category info if available
              if (data.category_info) {
                doc.metadata.predicted_category = data.category_info.predicted_category;
                doc.category = data.category_info.current_category || doc.category;
              }
            } else {
              console.error(`Error loading document ${doc.id}: ${response.statusText}`);
            }
          } catch (error) {
            console.error(`Error loading details for document ${doc.id}:`, error);
          }
        }));
      }
    } catch (error) {
      console.error('Error loading document details:', error);
    } finally {
      loadingDetails = false;
    }
  }

  onMount(async () => {
    try {
      // 1. Load basic document list
      const response = await fetch('/api/pdfs');
      documents = await response.json();

      // 2. Generate thumbnails
      for (const doc of documents) {
        await generateThumbnail(doc);
      }

      // 3. Load detailed document information with summaries
      await loadDocumentDetails();

    } catch (e) {
      error = 'Failed to load documents.';
    } finally {
      loading = false;
    }
  });

  async function handleRetriggerEmbedding(pdfId: string) {
    try {
      await triggerEmbeddingProcess(pdfId);
      alert(`Embedding process re-triggered for document ID: ${pdfId}`);
    } catch (e) {
      alert(`Failed to re-trigger embedding: ${e.message}`);
    }
  }

  async function handleDelete(pdfId: string) {
      if (confirm('Are you sure you want to delete this document?')) {
          try {
              await softDeleteDocument(pdfId);
              documents = documents.filter(doc => doc.id !== pdfId);
          } catch (e) {
              alert(`Failed to delete document: ${e.message}`);
          }
      }
  }
</script>

<div class="min-h-screen bg-gray-50 py-8">
  <div class="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
    <!-- Header -->
    <div class="flex justify-between items-center mb-8">
      <h1 class="text-3xl font-bold text-gray-900">Document Library</h1>
      <div class="flex items-center gap-4">
        <button
          class="bg-blue-100 hover:bg-blue-200 text-blue-700 px-4 py-2 rounded-lg flex items-center gap-2"
          on:click={() => showSummaries = !showSummaries}
          aria-label={showSummaries ? "Hide document summaries" : "Show document summaries"}
        >
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d={showSummaries ? "M13.875 18.825A10.05 10.05 0 0112 19c-4.478 0-8.268-2.943-9.543-7a9.97 9.97 0 011.563-3.029m5.858.908a3 3 0 114.243 4.243M9.878 9.878l4.242 4.242M9.88 9.88l-3.29-3.29m7.532 7.532l3.29 3.29M3 3l3.59 3.59m0 0A9.953 9.953 0 0112 5c4.478 0 8.268 2.943 9.543 7a10.025 10.025 0 01-4.132 5.411m0 0L21 21" : "M15 12a3 3 0 11-6 0 3 3 0 016 0z M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z"} />
          </svg>
          {showSummaries ? "Hide Summaries" : "Show Summaries"}
        </button>
        <button
          class="bg-blue-600 hover:bg-blue-700 text-white px-6 py-2 rounded-lg flex items-center gap-2 shadow-sm"
          on:click={() => goto('/documents/new')}
        >
          <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 4v16m8-8H4" />
          </svg>
          Add Document
        </button>
      </div>
    </div>

    <!-- Filters -->
    <div class="bg-white rounded-lg shadow-sm p-4 mb-6">
      <div class="flex flex-col sm:flex-row gap-4">
        <div class="flex-1">
          <input
            type="text"
            placeholder="Search documents..."
            class="w-full px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            bind:value={searchQuery}
          />
        </div>
        <div class="flex gap-4">
          <select
            class="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            bind:value={selectedCategory}
          >
            {#each categories as category}
              <option value={category.id}>{category.name}</option>
            {/each}
          </select>
          <select
            class="px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            bind:value={sortBy}
          >
            <option value="name">Sort by Name</option>
            <option value="date">Sort by Date</option>
          </select>
          <div class="flex border border-gray-300 rounded-lg overflow-hidden">
            <button
              class="px-4 py-2 {viewMode === 'grid' ? 'bg-blue-50 text-blue-600' : 'bg-white text-gray-600'}"
              on:click={() => viewMode = 'grid'}
              aria-label="Grid view"
            >
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2V6zM14 6a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2V6zM4 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2H6a2 2 0 01-2-2v-2zM14 16a2 2 0 012-2h2a2 2 0 012 2v2a2 2 0 01-2 2h-2a2 2 0 01-2-2v-2z" />
              </svg>
            </button>
            <button
              class="px-4 py-2 {viewMode === 'list' ? 'bg-blue-50 text-blue-600' : 'bg-white text-gray-600'}"
              on:click={() => viewMode = 'list'}
              aria-label="List view"
            >
              <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Document Grid -->
    {#if loading}
      <div class="flex justify-center items-center h-64">
        <div class="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
      </div>
    {:else if loadingDetails}
      <div class="flex flex-col items-center justify-center py-4">
        <div class="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600 mb-2"></div>
        <p class="text-gray-600">Loading document summaries...</p>
      </div>
    {:else if error}
      <div class="bg-red-50 text-red-600 p-4 rounded-lg">{error}</div>
    {:else if filteredDocuments.length === 0}
      <div class="text-center py-12">
        <svg class="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
        </svg>
        <h3 class="mt-2 text-sm font-medium text-gray-900">No documents found</h3>
        <p class="mt-1 text-sm text-gray-500">Try adjusting your search or filter criteria.</p>
      </div>
    {:else}
      <div class={viewMode === 'grid' ? 'grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6' : 'space-y-4'}>
        {#each filteredDocuments as doc}
          <div
            class={viewMode === 'grid' ? 'bg-white rounded-lg shadow-sm overflow-hidden hover:shadow-md transition-shadow' : 'bg-white rounded-lg shadow-sm overflow-hidden hover:shadow-md transition-shadow flex'}
            role="article"
          >
            <!-- Thumbnail -->
            <button
              class={viewMode === 'grid' ? 'aspect-[4/3] relative bg-gray-100 w-full' : 'w-48 relative bg-gray-100'}
              on:click={() => goto(`/documents/${doc.id}`)}
              on:keydown={(e) => e.key === 'Enter' && goto(`/documents/${doc.id}`)}
              aria-label={`View document: ${doc.name}`}
            >
              {#if doc.thumbnail}
                <img src={doc.thumbnail} alt={doc.name} class="w-full h-full object-cover" />
              {:else}
                <div class="absolute inset-0 flex items-center justify-center">
                  <svg class="w-12 h-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                  </svg>
                </div>
              {/if}
            </button>

            <!-- Content -->
            <div class="p-4 flex-1 flex flex-col">
              <div class="flex justify-between items-start">
                <div>
                  <h3 class="text-lg font-medium text-gray-900">
                    <button
                      class="text-left hover:underline focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 rounded"
                      on:click={() => goto(`/documents/${doc.id}`)}
                      on:keydown={(e) => e.key === 'Enter' && goto(`/documents/${doc.id}`)}
                    >
                      {doc.name}
                    </button>
                  </h3>
                  <p class="mt-1 text-sm text-gray-500">{doc.description || 'No description available'}</p>
                </div>
                <div class="flex gap-2">
                  <button
                    class="text-blue-600 hover:text-blue-800 focus:outline-none focus:ring-2 focus:ring-blue-500 p-1 rounded"
                    on:click={() => goto(`/documents/${doc.id}`)}
                    aria-label={`View ${doc.name}`}
                  >
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                    </svg>
                  </button>
                  <button
                    class="text-yellow-600 hover:text-yellow-800 focus:outline-none focus:ring-2 focus:ring-yellow-500 p-1 rounded"
                    on:click={() => handleRetriggerEmbedding(doc.id)}
                    aria-label={`Re-trigger processing for ${doc.name}`}
                  >
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                    </svg>
                  </button>
                  <button
                    class="text-red-600 hover:text-red-800 focus:outline-none focus:ring-2 focus:ring-red-500 p-1 rounded"
                    on:click={() => handleDelete(doc.id)}
                    aria-label={`Delete ${doc.name}`}
                  >
                    <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                  </button>
                </div>
              </div>

              <div class="mt-2 flex items-center gap-2">
                <span class="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-blue-100 text-blue-800">
                  {doc.category || 'Uncategorized'}
                </span>
                <span class="text-xs text-gray-500">
                  Updated {new Date(doc.updatedAt || Date.now()).toLocaleDateString()}
                </span>
              </div>

              <!-- Document Summary Section (using the new component) -->
              {#if showSummaries}
                <div class="mt-3 border-t border-gray-100 pt-3">
                  <!-- Added debugging info to show what's available -->
                  {#if doc.metadata?.document_summary}
                    <DocumentSummary
                      summary={doc.metadata.document_summary}
                      categoryInfo={{
                        current_category: doc.category || 'general',
                        predicted_category: doc.metadata?.predicted_category
                      }}
                      truncateLength={60}
                      compact={true}
                    />
                  {:else}
                    <p class="text-xs text-gray-400 italic">
                      Summary not available. Try re-processing the document.
                    </p>
                  {/if}
                </div>
              {/if}
            </div>
          </div>
        {/each}
      </div>
    {/if}
  </div>
</div>

<style>
  :global(body) {
    background-color: #f9fafb;
  }
</style>
