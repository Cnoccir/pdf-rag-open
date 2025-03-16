<script lang="ts">
import { onMount, onDestroy } from 'svelte';
import * as pdfjs from 'pdfjs-dist';
import { ChevronLeft, ChevronRight, Search, X, ChevronUp, ChevronDown, List, Plus, Minus } from 'lucide-svelte';
import { debounce } from 'lodash-es';
import {
    smoothScrollToPage,
    setupZoom,
    improveSearchHighlighting,
    debouncedScrollHandler,
    enhanceTextSelection,
    updateCurrentPage
} from './pdfViewerEnhancements.js';
import PDFWorker from 'pdfjs-dist/build/pdf.worker.min.js?url';
pdfjs.GlobalWorkerOptions.workerSrc = PDFWorker;

export let url = '';
export let scrollToPage: string | null = null;
export let scrollToElementId: string | null = null;

let zoomControls;
let currentScale = 1;
let canvasContainer: HTMLDivElement;
let error: string | null = null;
let currentPage = 1;
let totalPages = 0;
let pdfDocument: pdfjs.PDFDocumentProxy | null = null;
let searchResults: Array<SearchResult> = [];
let displayedResults: Array<SearchResult> = [];
let searchQuery = '';
let destroyed = false;
let isSearchBarVisible = false;
let isResultListVisible = false;
let currentSearchIndex = 0;
let isLoading = true;
let isSearching = false;
let selectedResultIndex = -1;
let currentResultsPage = 1;
const resultsPerPage = 10;
let searchCancelled = false;
let renderedPages: Set<number> = new Set();
let isSearchMinimized = false;
let manualPageInput = '';
let pagePositions: number[] = [];

interface SearchResult {
    page: number;
    text: string;
    snippet: string;
    position: {
        start: number;
        end: number;
    };
}

$: canGoToPrevPage = currentPage > 1;
$: canGoToNextPage = currentPage < totalPages;
$: zoomPercentage = zoomControls ? Math.round(zoomControls.getCurrentScale() * 100) : 100;
$: if (canvasContainer && pdfDocument && !isLoading) {
    handleScroll();
}

$: if (isResultListVisible && displayedResults.length > 0 && selectedResultIndex === -1) {
    selectedResultIndex = 0;
}
const debouncedSearch = debounce(async () => {
    if (searchQuery.length < 2) {
        searchResults = [];
        updateDisplayedResults();
        isResultListVisible = false;
        return;
    }
    isSearching = true;
    searchResults = [];
    searchCancelled = false;
    currentResultsPage = 1;
    const uniqueResults = new Set<string>();

    for (let i = 1; i <= pdfDocument!.numPages; i++) {
        if (searchCancelled) break;

        const page = await pdfDocument!.getPage(i);
        const textContent = await page.getTextContent();
        const text = textContent.items.map(item => 'str' in item ? item.str : '').join(' ');
        const regex = new RegExp(searchQuery, 'gi');
        let match;

        while ((match = regex.exec(text)) !== null) {
            if (searchCancelled) break;

            const start = Math.max(0, match.index - 40);
            const end = Math.min(text.length, match.index + match[0].length + 40);
            const snippet = text.slice(start, end);
            const resultKey = `${i}-${match.index}`;

            if (!uniqueResults.has(resultKey)) {
                uniqueResults.add(resultKey);
                searchResults.push({
                    page: i,
                    text: match[0],
                    snippet: snippet.replace(regex, '<mark>$&</mark>'),
                    position: { start: match.index, end: match.index + match[0].length }
                });

                if (searchResults.length % 10 === 0) {
                    updateDisplayedResults();
                    await new Promise(resolve => setTimeout(resolve, 0)); // Allow UI to update
                }
            }
        }
    }

    updateDisplayedResults();
    isSearching = false;
    isResultListVisible = searchResults.length > 0;
}, 300);

function updateDisplayedResults() {
    const startIndex = (currentResultsPage - 1) * resultsPerPage;
    displayedResults = searchResults.slice(startIndex, startIndex + resultsPerPage);
}

function nextResultsPage() {
    if (currentResultsPage * resultsPerPage < searchResults.length) {
        currentResultsPage++;
        updateDisplayedResults();
    }
}

function prevResultsPage() {
    if (currentResultsPage > 1) {
        currentResultsPage--;
        updateDisplayedResults();
    }
}

function getCurrentPage() {
    const scrollTop = canvasContainer.scrollTop;
    const containerHeight = canvasContainer.clientHeight;
    const middlePosition = scrollTop + containerHeight / 2;
    const pageIndex = pagePositions.findIndex(position => position > middlePosition) - 1;
    return Math.max(1, Math.min(pageIndex, totalPages));
}

function handleScroll() {
    if (!canvasContainer || !pdfDocument) return;

    const scrollTop = canvasContainer.scrollTop;
    const containerHeight = canvasContainer.clientHeight;
    const scrollBottom = scrollTop + containerHeight;

    const newCurrentPage = getCurrentPage();
    if (newCurrentPage !== currentPage) {
        currentPage = newCurrentPage;
        dispatchEvent(new CustomEvent('pagechange', { detail: { page: currentPage } }));
    }

    const visiblePages = getVisiblePages(scrollTop, scrollBottom);
    for (let i = visiblePages.first; i <= visiblePages.last; i++) {
        if (!renderedPages.has(i)) {
            renderPage(i);
        }
    }

    // Log for debugging
    console.log(`Current page: ${currentPage}, Scroll position: ${scrollTop}, Page positions:`, pagePositions);
}

function getVisiblePages(scrollTop, scrollBottom) {
    let first = totalPages + 1;
    let last = 0;

    pagePositions.forEach((position, index) => {
        if (position < scrollBottom && pagePositions[index + 1] > scrollTop) {
            first = Math.min(first, index + 1);
            last = Math.max(last, index + 1);
        }
    });

    return { first, last };
}

function enhanceTextLayer(textLayer, viewport) {
    textLayer.style.left = '0';
    textLayer.style.top = '0';
    textLayer.style.right = '0';
    textLayer.style.bottom = '0';
    textLayer.style.position = 'absolute';
    textLayer.setAttribute('data-main-rotation', viewport.rotation);
}

async function renderPage(pageNumber: number) {
    if (renderedPages.has(pageNumber) || pageNumber < 1 || pageNumber > totalPages) return;

    console.log(`Rendering page ${pageNumber}`);  // Add this log

    const page = await pdfDocument!.getPage(pageNumber);
    const scale = currentScale * 2.0;
    const viewport = page.getViewport({ scale: scale });
    const wrapper = document.createElement('div');
    wrapper.className = 'page-wrapper mb-4 relative w-full';
    wrapper.id = `page-${pageNumber}`;

    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    canvas.height = viewport.height;
    canvas.width = viewport.width;
    canvas.style.width = '100%';
    wrapper.appendChild(canvas);

    const textLayerDiv = document.createElement('div');
    textLayerDiv.className = 'textLayer';
    enhanceTextLayer(textLayerDiv, viewport);
    wrapper.appendChild(textLayerDiv);

    canvasContainer.appendChild(wrapper);

    const renderContext = {
        canvasContext: ctx,
        viewport: viewport
    };
    await page.render(renderContext).promise;

    const textContent = await page.getTextContent();
    const textLayerViewport = viewport.clone({ dontFlip: true });
    pdfjs.renderTextLayer({
        textContent: textContent,
        container: textLayerDiv,
        viewport: viewport,
        textDivs: []
    });

    enhanceTextSelection(textLayerDiv);

    renderedPages.add(pageNumber);
    calculatePagePositions();  // Recalculate page positions after rendering
    console.log(`Page ${pageNumber} rendered`);
}

onMount(async () => {
    if (!url) {
        error = "No URL provided for PDF";
        isLoading = false;
        return;
    }

    try {
        const loadingTask = pdfjs.getDocument(url);
        pdfDocument = await loadingTask.promise;
        if (destroyed) return;

        totalPages = pdfDocument.numPages;
        await new Promise(resolve => setTimeout(resolve, 0));

        if (canvasContainer) {
            zoomControls = setupZoom(currentScale);

            await renderInitialPages();
            calculatePagePositions();
            currentPage = Math.max(1, getCurrentPage());  // Ensure it's never less than 1

            canvasContainer.addEventListener('scroll', handleScroll);

            // Set initial scroll position to top
            canvasContainer.scrollTop = 0;
            handleScroll();

            if (scrollToPage !== null) {
                const pageNumber = parseInt(scrollToPage, 10);
                if (!isNaN(pageNumber) && pageNumber > 0 && pageNumber <= totalPages) {
                    await scrollTo(pageNumber);
                    if (scrollToElementId) {
                        setTimeout(() => {
                            const element = document.getElementById(scrollToElementId);
                            if (element) {
                                element.scrollIntoView({ behavior: 'smooth', block: 'center' });
                                element.classList.add('highlight-element');
                            }
                        }, 500);
                    }
                }
            }
        }
    } catch (e) {
        console.error("Error loading PDF:", e);
        error = `Failed to load PDF: ${e.message}`;
    } finally {
        isLoading = false;
    }
});

async function renderInitialPages() {
    for (let i = 1; i <= Math.min(5, totalPages); i++) {
        await renderPage(i);
    }
}

onDestroy(() => {
    destroyed = true;
    if (canvasContainer) {
        canvasContainer.removeEventListener('scroll', handleScroll);
    }
});

async function scrollTo(pageNumber: number) {
    if (pageNumber > 0 && pageNumber <= totalPages) {
        // Ensure the target page and surrounding pages are rendered
        for (let i = Math.max(1, pageNumber - 1); i <= Math.min(totalPages, pageNumber + 1); i++) {
            await renderPage(i);
        }

        const approximatePageHeight = canvasContainer.scrollHeight / totalPages;
        const targetScrollTop = (pageNumber - 1) * approximatePageHeight;
        canvasContainer.scrollTo({
            top: targetScrollTop,
            behavior: 'smooth'
        });

        currentPage = pageNumber;
    }
}

async function scrollToResult(result: SearchResult) {
    await scrollTo(result.page);
    await new Promise(resolve => setTimeout(resolve, 100));
    highlightSearchResult(result);
    minimizeSearch();
}

function minimizeSearch() {
    isSearchMinimized = true;
    isSearchBarVisible = false;
}

function highlightSearchResult(result: SearchResult) {
    const pageElement = document.getElementById(`page-${result.page}`);
    if (!pageElement) return;

    const textLayer = pageElement.querySelector('.textLayer');
    if (!textLayer) return;

    improveSearchHighlighting(textLayer, result.text);

    const highlight = pageElement.querySelector('.enhanced-highlight');
    if (highlight) {
        highlight.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}


function clearSearch() {
    searchCancelled = true;
    searchResults = [];
    displayedResults = [];
    searchQuery = '';
    currentSearchIndex = 0;
    isSearchBarVisible = false;
    isResultListVisible = false;
    isSearchMinimized = false;
    const highlights = document.querySelectorAll('.search-highlight');
    highlights.forEach(h => {
        const parent = h.parentNode;
        if (parent) {
            parent.replaceChild(document.createTextNode(h.textContent || ''), h);
            parent.normalize();
        }
    });
}

function handleKeydown(event: KeyboardEvent, index: number) {
    if (event.key === 'Enter') {
        currentSearchIndex = index;
        scrollToResult(displayedResults[index]);
    } else if (event.key === 'ArrowDown') {
        event.preventDefault();
        selectedResultIndex = Math.min(selectedResultIndex + 1, displayedResults.length - 1);
    } else if (event.key === 'ArrowUp') {
        event.preventDefault();
        selectedResultIndex = Math.max(selectedResultIndex - 1, 0);
    }
}

function toggleSearchBar() {
    if (isSearchMinimized) {
        isSearchMinimized = false;
        isSearchBarVisible = true;
        isResultListVisible = searchResults.length > 0;
    } else {
        isSearchBarVisible = !isSearchBarVisible;
        if (!isSearchBarVisible) {
            isResultListVisible = false;
        } else if (searchResults.length > 0) {
            isResultListVisible = true;
        }
    }
}

function handleManualPageInput() {
    const pageNumber = parseInt(manualPageInput, 10);
    if (!isNaN(pageNumber) && pageNumber > 0 && pageNumber <= totalPages) {
        scrollTo(pageNumber);
    }
    manualPageInput = '';
}

function nextPage() {
    if (canGoToNextPage) scrollTo(currentPage + 1);
}

function previousPage() {
    if (canGoToPrevPage) scrollTo(currentPage - 1);
}

function calculatePagePositions() {
    pagePositions = [0];
    const pages = canvasContainer.querySelectorAll('.page-wrapper');
    let totalHeight = 0;
    pages.forEach((page) => {
        totalHeight += page.clientHeight;
        pagePositions.push(totalHeight);
    });
    // Ensure there's always at least one page
    if (pagePositions.length < 2) {
        pagePositions.push(canvasContainer.clientHeight);
    }
    console.log('Page positions calculated:', pagePositions);
}

function zoomIn() {
    zoomControls.zoomIn();
    currentScale = zoomControls.getCurrentScale();
}

function zoomOut() {
    zoomControls.zoomOut();
    currentScale = zoomControls.getCurrentScale();
}

function resetZoom() {
    zoomControls.resetZoom();
    currentScale = zoomControls.getCurrentScale();
}
</script>
<div class="relative w-full h-screen bg-gray-100">
    <div class="absolute top-4 left-1/2 transform -translate-x-1/2 z-10 bg-white rounded-lg shadow-md p-2 flex items-center space-x-2">
        <button on:click={previousPage} disabled={!canGoToPrevPage} class="p-1 hover:bg-gray-100 rounded disabled:opacity-50">
            <ChevronLeft class="w-5 h-5" />
        </button>
        <span class="text-sm">Page {currentPage} of {totalPages}</span>
        <button on:click={nextPage} disabled={!canGoToNextPage} class="p-1 hover:bg-gray-100 rounded disabled:opacity-50">
            <ChevronRight class="w-5 h-5" />
        </button>
        <input
            type="text"
            placeholder="Go to page"
            bind:value={manualPageInput}
            on:keydown={(e) => e.key === 'Enter' && handleManualPageInput()}
            class="w-20 text-center border rounded px-1 py-0.5 text-sm"
        />
        <div class="h-6 w-px bg-gray-300 mx-1"></div>
        <button on:click={zoomOut} class="p-1 hover:bg-gray-100 rounded">
            <Minus class="w-5 h-5" />
        </button>
        <span class="text-sm">{Math.round(currentScale * 100)}%</span>
        <button on:click={zoomIn} class="p-1 hover:bg-gray-100 rounded">
            <Plus class="w-5 h-5" />
        </button>
        <button on:click={resetZoom} class="text-xs bg-gray-100 hover:bg-gray-200 rounded px-2 py-1">
            Reset
        </button>
        <div class="h-6 w-px bg-gray-300 mx-1"></div>
        <button on:click={toggleSearchBar} class="p-1 hover:bg-gray-100 rounded">
            <Search class="w-5 h-5" />
        </button>
    </div>

    {#if isSearchBarVisible || isSearchMinimized}
        <div class="search-ui absolute top-16 left-1/2 transform -translate-x-1/2 z-20 bg-white rounded-lg shadow-md p-2 flex items-center space-x-2 {isSearchMinimized ? 'search-minimized' : ''}">
            {#if !isSearchMinimized}
                <input
                    type="text"
                    placeholder="Search..."
                    bind:value={searchQuery}
                    on:input={debouncedSearch}
                    class="border rounded px-2 py-1 text-sm"
                />
                <button on:click={() => isResultListVisible = !isResultListVisible} class="p-1 hover:bg-gray-100 rounded">
                    <List class="w-4 h-4" />
                </button>
                <span class="text-xs">{searchResults.length} results</span>
                <button on:click={clearSearch} class="p-1 hover:bg-gray-100 rounded">
                    <X class="w-4 h-4" />
                </button>
            {/if}
            <button on:click={() => isSearchMinimized = !isSearchMinimized} class="p-1 hover:bg-gray-100 rounded">
                {#if isSearchMinimized}
                    <ChevronDown class="w-4 h-4" />
                {:else}
                    <ChevronUp class="w-4 h-4" />
                {/if}
            </button>
        </div>
    {/if}

    {#if isResultListVisible && !isSearchMinimized && displayedResults.length > 0}
        <div class="result-list absolute top-28 left-1/2 transform -translate-x-1/2 z-20 bg-white rounded-lg shadow-md p-2 max-h-60 overflow-y-auto w-96">
            {#each displayedResults as result, index}
                <div
                    role="button"
                    tabindex="0"
                    class="p-2 hover:bg-gray-100 cursor-pointer {index === currentSearchIndex ? 'bg-blue-100' : ''} {index === selectedResultIndex ? 'ring-2 ring-blue-500' : ''}"
                    on:click={() => {
                        currentSearchIndex = index;
                        scrollToResult(result);
                    }}
                    on:keydown={(e) => handleKeydown(e, index)}
                >
                    <p class="text-sm font-semibold">Page {result.page}</p>
                    <p class="text-xs">{@html result.snippet}</p>
                </div>
            {/each}
            <div class="flex justify-between mt-2">
                <button on:click={prevResultsPage} disabled={currentResultsPage === 1} class="text-sm px-2 py-1 bg-gray-200 rounded">Previous</button>
                <span class="text-sm">{currentResultsPage} / {Math.ceil(searchResults.length / resultsPerPage)}</span>
                <button on:click={nextResultsPage} disabled={currentResultsPage * resultsPerPage >= searchResults.length} class="text-sm px-2 py-1 bg-gray-200 rounded">Next</button>
            </div>
        </div>
    {/if}

    {#if isSearching}
        <div class="absolute top-28 left-1/2 transform -translate-x-1/2 z-20 bg-white rounded-lg shadow-md p-2">
            Searching... (Found {searchResults.length} so far)
            <button on:click={() => searchCancelled = true} class="ml-2 text-red-500">Cancel</button>
        </div>
    {/if}

    <div class="pdf-container h-full flex justify-center items-center overflow-hidden relative">
        {#if error}
            <div class="error text-red-500">{error}</div>
        {:else if isLoading}
            <div class="loading-overlay absolute inset-0 flex items-center justify-center bg-white bg-opacity-90">
                <div class="animate-spin rounded-full h-32 w-32 border-t-2 border-b-2 border-gray-900"></div>
                <span class="ml-3">Loading PDF...</span>
            </div>
        {:else}
            <div bind:this={canvasContainer} class="pdf-wrapper w-full h-full overflow-auto bg-gray-200 p-4"></div>
        {/if}
    </div>
</div>
<style>
    :global(.pdf-wrapper canvas) {
        max-width: 100%;
        height: auto;
    }
    :global(.highlight-element) {
        background-color: yellow;
        transition: background-color 0.3s ease;
    }
    :global(.textLayer) {
        position: absolute;
        left: 0;
        top: 0;
        right: 0;
        bottom: 0;
        overflow: hidden;
        opacity: 1;
        line-height: 1.0;
        text-align: initial;
        pointer-events: auto;
        user-select: text;
    }

    :global(.textLayer > span) {
        color: transparent;
        position: absolute;
        white-space: pre;
        cursor: text;
        transform-origin: 0% 0%;
    }
    :global(.textLayer .highlight) {
        margin: -1px;
        padding: 1px;
        background-color: rgb(180, 0, 170);
        border-radius: 4px;
    }
    :global(.textLayer .highlight.begin) {
        border-radius: 4px 0px 0px 4px;
    }
    :global(.textLayer .highlight.end) {
        border-radius: 0px 4px 4px 0px;
    }
    :global(.textLayer .highlight.middle) {
        border-radius: 0px;
    }
    :global(.textLayer .highlight.selected) {
        background-color: rgb(0, 100, 0);
    }
    :global(.search-highlight) {
        background-color: yellow;
        color: black;
        border-radius: 2px;
        padding: 0 2px;
    }
    .search-ui {
        transition: all 0.3s ease;
    }
    .search-ui.search-minimized {
        transform: translate(-50%, -100%);
        top: 4rem;
    }
    .result-list {
        max-height: 60vh;
        overflow-y: auto;
    }
    :global(.enhanced-highlight) {
        background-color: yellow;
        color: black;
        border-radius: 2px;
        padding: 0 2px;
    }
</style>
