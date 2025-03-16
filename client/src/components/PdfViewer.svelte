<!-- PdfViewer.svelte -->
<script lang="ts">
  /*****************************************
   * IMPORTS
   *****************************************/
  import { onMount, onDestroy, createEventDispatcher } from 'svelte';
  import * as pdfjs from 'pdfjs-dist';
  import PDFWorker from 'pdfjs-dist/build/pdf.worker.min.js?url';
  import { debounce } from 'lodash-es';
  import Icon from '$c/Icon.svelte';

  // Set up PDF.js worker
  pdfjs.GlobalWorkerOptions.workerSrc = PDFWorker;

  /*****************************************
   * EXPORTED PROPS
   *****************************************/
  export let url = '';
  export let scrollToPage = '1';
  export let scrollToElementId: string | null = null;
  export const contextSections = [];

  /*****************************************
   * COMPONENT STATE
   *****************************************/
  let canvasContainer: HTMLDivElement;
  let pdfDocument: pdfjs.PDFDocumentProxy | null = null;
  let destroyed = false;
  let isLoading = true;
  let error: string | null = null;

  let currentPage = 1;
  let totalPages = 0;
  let currentScale = 1;
  let zoomPercentage = 100;
  let manualPageInput = '';

  let renderedPages: Set<number> = new Set();
  let pagePositions: number[] = [];
  let debounceTextLayerRender: any;

  // Keep track of pages currently rendering to avoid concurrency conflicts
  const inFlightRenders = new Map<number, boolean>();

  // Searching
  interface SearchResult {
    page: number;
    text: string;
    snippet: string;
    position: {
      start: number;
      end: number;
    };
  }

  let searchQuery = '';
  let searchResults: SearchResult[] = [];
  let displayedResults: SearchResult[] = [];
  let isSearching = false;
  let searchCancelled = false;
  let isSearchBarVisible = false;
  let isSearchMinimized = false;
  let isResultListVisible = false;
  let currentSearchIndex = 0;
  let selectedResultIndex = -1;
  let currentResultsPage = 1;
  const resultsPerPage = 10;

  // Viewport monitoring
  let isMobileView = false;
  let documentWidth = 0;
  let windowWidth = 0;
  let containerObserver: ResizeObserver | null = null;

  // Event dispatcher
  const dispatch = createEventDispatcher();

  /*****************************************
   * COMPUTED STATE
   *****************************************/
  $: canGoToPrevPage = currentPage > 1;
  $: canGoToNextPage = currentPage < totalPages;
  $: zoomPercentage = Math.round(currentScale * 100);

  $: if (scrollToPage && pdfDocument) {
    const page = parseInt(scrollToPage);
    if (!isNaN(page) && page >= 1 && page <= totalPages) {
      scrollTo(page);
    }
  }

  $: if (scrollToElementId && pdfDocument) {
    highlightElementById(scrollToElementId);
  }

  /*****************************************
   * DEBOUNCED SEARCH
   *****************************************/
  const debouncedSearch = debounce(async () => {
    await performSearch();
  }, 300);

  async function performSearch() {
    if (!pdfDocument) return;
    if (searchQuery.length < 2) {
      searchResults = [];
      updateDisplayedResults();
      isResultListVisible = false;
      return;
    }

    isSearching = true;
    searchCancelled = false;
    searchResults = [];
    currentResultsPage = 1;

    const uniqueResults = new Set<string>();

    // Enhanced search with page rendering support
    for (let i = 1; i <= pdfDocument.numPages; i++) {
      if (searchCancelled) break;

      // Check if the page is already rendered, if not, prioritize rendering it
      if (!renderedPages.has(i)) {
        await renderPage(i);
      }

      const page = await pdfDocument.getPage(i);
      const textContent = await page.getTextContent();
      const pageText = textContent.items.map(item => ('str' in item ? item.str : '')).join(' ');

      const regex = new RegExp(searchQuery, 'gi');
      let match;
      while ((match = regex.exec(pageText)) !== null) {
        if (searchCancelled) break;

        const start = Math.max(0, match.index - 40);
        const end = Math.min(pageText.length, match.index + match[0].length + 40);
        const snippet = pageText.slice(start, end);
        const resultKey = `${i}-${match.index}`;

        if (!uniqueResults.has(resultKey)) {
          uniqueResults.add(resultKey);
          searchResults.push({
            page: i,
            text: match[0],
            snippet: snippet.replace(
              new RegExp(`(${searchQuery})`, 'gi'),
              '<mark>$1</mark>'
            ),
            position: { start: match.index, end: match.index + match[0].length }
          });

          // Yield to UI every 10 hits
          if (searchResults.length % 10 === 0) {
            updateDisplayedResults();
            await new Promise(resolve => setTimeout(resolve, 0));
          }
        }
      }
    }

    updateDisplayedResults();
    isSearching = false;
    isResultListVisible = searchResults.length > 0;

    // If we have results, highlight the first one
    if (searchResults.length > 0) {
      currentSearchIndex = 0;
      await scrollToResult(searchResults[0]);
    }
  }

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

  async function clearSearch() {
    searchCancelled = true;
    searchQuery = '';
    searchResults = [];
    displayedResults = [];
    currentSearchIndex = 0;
    isSearchBarVisible = false;
    isResultListVisible = false;
    isSearchMinimized = false;

    // Clear highlighting
    clearHighlights();
  }

  function clearHighlights() {
    // Remove enhanced highlights from text layer
    const highlights = document.querySelectorAll('.enhanced-highlight');
    highlights.forEach(highlight => {
      if (highlight.parentNode) {
        const text = document.createTextNode(highlight.textContent || '');
        highlight.parentNode.replaceChild(text, highlight);
      }
    });

    // Remove any temporary highlight spans
    const tempHighlights = document.querySelectorAll('.highlight-search');
    tempHighlights.forEach(highlight => {
      highlight.classList.remove('highlight-search');
    });
  }

  /*****************************************
   * PAGE POSITION CALCULATIONS
   *****************************************/
  function calculatePagePositions() {
    if (!canvasContainer) return;
    const pageWrappers = canvasContainer.querySelectorAll('.page-wrapper');

    let cumulativeHeight = 0;
    pagePositions = [0];

    pageWrappers.forEach(wrapper => {
      cumulativeHeight += wrapper.clientHeight;
      pagePositions.push(cumulativeHeight);
    });

    // Estimate positions for any unrendered pages
    if (pageWrappers.length > 0 && pageWrappers.length < totalPages) {
      const avgHeight = cumulativeHeight / pageWrappers.length;
      while (pagePositions.length <= totalPages) {
        cumulativeHeight += avgHeight;
        pagePositions.push(cumulativeHeight);
      }
    }
  }

  function getCurrentPage(): number {
    if (!canvasContainer || pagePositions.length <= 1) return 1;
    const midpoint = canvasContainer.scrollTop + (canvasContainer.clientHeight / 2);

    for (let i = 1; i < pagePositions.length; i++) {
      if (midpoint < pagePositions[i]) {
        return i;
      }
    }
    return totalPages;
  }

  function getVisiblePages() {
    if (!canvasContainer) return { first: 1, last: 1 };

    const scrollTop = canvasContainer.scrollTop;
    const scrollBottom = scrollTop + canvasContainer.clientHeight;

    let first = totalPages;
    let last = 1;

    for (let i = 1; i < pagePositions.length; i++) {
      const pageTop = pagePositions[i - 1];
      const pageBottom = pagePositions[i];
      if (pageBottom >= scrollTop && pageTop <= scrollBottom) {
        first = Math.min(first, i);
        last = Math.max(last, i);
      }
    }

    return { first, last };
  }

  const handleScroll = debounce(() => {
    if (!canvasContainer || !pdfDocument) return;

    const newPage = getCurrentPage();
    if (newPage !== currentPage) {
      currentPage = newPage;
      dispatch('pagechange', { page: currentPage });
    }

    const { first, last } = getVisiblePages();

    // Render pages in that visible range, plus a small buffer
    const buffer = 2;  // Render 2 pages before and after visible area
    const renderFrom = Math.max(1, first - buffer);
    const renderTo = Math.min(totalPages, last + buffer);

    const pagesToRender: number[] = [];
    for (let p = renderFrom; p <= renderTo; p++) {
      if (!renderedPages.has(p) && !inFlightRenders.get(p)) {
        pagesToRender.push(p);
      }
    }

    if (pagesToRender.length > 0) {
      Promise.all(pagesToRender.map(num => renderPage(num)))
        .then(() => {
          calculatePagePositions();
        });
    }
  }, 50);

  /*****************************************
   * CORE RENDERING
   *****************************************/
  async function renderPage(pageNumber: number) {
    if (!pdfDocument || pageNumber < 1 || pageNumber > totalPages) return;

    // If this page is currently rendering, skip
    if (inFlightRenders.get(pageNumber)) {
      return;
    }
    inFlightRenders.set(pageNumber, true);

    try {
      const existingWrapper = document.getElementById(`page-${pageNumber}`);
      if (existingWrapper) {
        // Already in DOM, so update it for the latest scale
        await updatePageRender(pageNumber, existingWrapper as HTMLElement);
        return;
      }

      // --- Create new DOM elements ---
      const page = await pdfDocument.getPage(pageNumber);

      // Combine rotation + scaling with high DPI support
      const baseRotation = page.rotation || 0;
      const devicePixelRatio = window.devicePixelRatio || 1;
      // Multiply by devicePixelRatio for crisp rendering
      const scaleFactor = currentScale * devicePixelRatio;

      const viewport = page.getViewport({ scale: scaleFactor, rotation: baseRotation });

      // Wrapper div - improved styling with max-width for better readability
      const wrapper = document.createElement('div');
      wrapper.id = `page-${pageNumber}`;
      wrapper.className = 'page-wrapper relative mb-8 flex justify-center';
      wrapper.setAttribute('data-page-number', pageNumber.toString());

      // Page container - helps with centering and adds a max-width
      const pageContainer = document.createElement('div');
      pageContainer.className = 'page-container relative shadow-lg rounded-lg overflow-hidden bg-white';
      pageContainer.style.width = `${viewport.width / devicePixelRatio}px`;
      pageContainer.style.height = `${viewport.height / devicePixelRatio}px`;

      // Get container width and set appropriate max-width
      const containerWidth = canvasContainer?.clientWidth || 1000;
      const maxWidth = Math.min(1000, containerWidth - 40); // 20px padding on each side
      pageContainer.style.maxWidth = `${maxWidth}px`;

      // Page number indicator
      const pageNumberIndicator = document.createElement('div');
      pageNumberIndicator.className = 'absolute bottom-3 right-3 bg-gray-800 bg-opacity-70 text-white text-xs py-1 px-2 rounded z-10';
      pageNumberIndicator.textContent = pageNumber.toString();

      // Canvas with proper pixel ratio handling
      const canvas = document.createElement('canvas');
      canvas.className = 'pdf-canvas w-full h-full';
      const ctx = canvas.getContext('2d', { alpha: false });
      if (!ctx) {
        throw new Error('Unable to get canvas context');
      }
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      canvas.style.width = `${viewport.width / devicePixelRatio}px`;
      canvas.style.height = `${viewport.height / devicePixelRatio}px`;

      // Text layer with improved styling and absolute positioning
      const textLayerDiv = document.createElement('div');
      textLayerDiv.className = 'text-layer absolute inset-0';
      textLayerDiv.style.width = `${viewport.width / devicePixelRatio}px`;
      textLayerDiv.style.height = `${viewport.height / devicePixelRatio}px`;

      // Assemble the DOM elements
      pageContainer.appendChild(canvas);
      pageContainer.appendChild(textLayerDiv);
      pageContainer.appendChild(pageNumberIndicator);
      wrapper.appendChild(pageContainer);

      // Insert wrapper into container in the correct order
      if (!canvasContainer) return;
      const existingPages = Array.from(canvasContainer.children);
      const nextPage = existingPages.find(p => parseInt(p.id.split('-')[1]) > pageNumber);
      if (nextPage) {
        canvasContainer.insertBefore(wrapper, nextPage);
      } else {
        canvasContainer.appendChild(wrapper);
      }

      // Render the PDF page to canvas with higher quality
      const renderContext = {
        canvasContext: ctx,
        viewport,
        enableWebGL: true,
        renderInteractiveForms: true
      };

      // Ensure we clear the canvas before rendering
      ctx.fillStyle = 'rgb(255, 255, 255)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      await page.render(renderContext).promise;

      // Render text layer with correct scaling
      const textContent = await page.getTextContent();
      try {
        await pdfjs.renderTextLayer({
          textContentSource: textContent,
          container: textLayerDiv,
          viewport: viewport.clone({ scale: 1 / devicePixelRatio }),
          textDivs: []
        }).promise;

        // Make text layer selectable after rendering
        setTimeout(() => {
          textLayerDiv.querySelectorAll('span').forEach((span, index) => {
            span.setAttribute('data-element-id', `${pageNumber}-${index}`);
            span.classList.add('text-element');

            // Prevent odd double-click behavior
            span.addEventListener('mousedown', (e) => {
              if (e.detail > 1) {
                e.preventDefault();
              }
            });
          });
        }, 10);
      } catch (err) {
        console.error('Error rendering text layer:', err);
      }

      // Mark as rendered
      renderedPages.add(pageNumber);

      // If this page is the current search result, highlight it
      if (searchResults.length > 0 && searchResults[currentSearchIndex]?.page === pageNumber) {
        highlightSearchResult(searchResults[currentSearchIndex]);
      }

    } catch (err) {
      console.error(`Error rendering page ${pageNumber}:`, err);
    } finally {
      inFlightRenders.delete(pageNumber);
    }
  }

  async function updatePageRender(pageNumber: number, wrapperElement: HTMLElement) {
    if (!pdfDocument) return;

    // If we are already updating this page, skip
    if (inFlightRenders.get(pageNumber)) {
      return;
    }
    inFlightRenders.set(pageNumber, true);

    try {
      const page = await pdfDocument.getPage(pageNumber);
      const baseRotation = page.rotation || 0;
      const devicePixelRatio = window.devicePixelRatio || 1;
      const scaleFactor = currentScale * devicePixelRatio;
      const viewport = page.getViewport({ scale: scaleFactor, rotation: baseRotation });

      // Update page container dimensions
      const pageContainer = wrapperElement.querySelector('.page-container');
      if (pageContainer) {
        const containerWidth = canvasContainer?.clientWidth || 1000;
        const maxWidth = Math.min(1000, containerWidth - 40); // 20px padding on each side

        pageContainer.setAttribute('style', `
          width: ${viewport.width / devicePixelRatio}px;
          height: ${viewport.height / devicePixelRatio}px;
          max-width: ${maxWidth}px;
        `);
      }

      // Update canvas
      const canvas = wrapperElement.querySelector('canvas');
      if (!canvas) return;
      const ctx = canvas.getContext('2d', { alpha: false });
      if (!ctx) return;

      // Resize canvas
      canvas.width = viewport.width;
      canvas.height = viewport.height;
      canvas.style.width = `${viewport.width / devicePixelRatio}px`;
      canvas.style.height = `${viewport.height / devicePixelRatio}px`;

      // Clear canvas
      ctx.fillStyle = 'rgb(255, 255, 255)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);

      // Render PDF
      await page.render({
        canvasContext: ctx,
        viewport,
        enableWebGL: true,
        renderInteractiveForms: true
      }).promise;

      // Re-render text layer
      const textLayerDiv = wrapperElement.querySelector('.text-layer');
      if (!textLayerDiv) return;
      textLayerDiv.innerHTML = ''; // clear old text

      textLayerDiv.style.width = `${viewport.width / devicePixelRatio}px`;
      textLayerDiv.style.height = `${viewport.height / devicePixelRatio}px`;

      const textContent = await page.getTextContent();
      try {
        await pdfjs.renderTextLayer({
          textContentSource: textContent,
          container: textLayerDiv,
          viewport: viewport.clone({ scale: 1 / devicePixelRatio }),
          textDivs: []
        }).promise;

        // Re-apply element IDs and improve text selection
        setTimeout(() => {
          textLayerDiv.querySelectorAll('span').forEach((span, index) => {
            span.setAttribute('data-element-id', `${pageNumber}-${index}`);
            span.classList.add('text-element');

            // Prevent odd double-click behavior
            span.addEventListener('mousedown', (e) => {
              if (e.detail > 1) {
                e.preventDefault();
              }
            });
          });
        }, 10);
      } catch (err) {
        console.error('Error updating text layer:', err);
      }

    } catch (err) {
      console.error(`Error updating page ${pageNumber}:`, err);
    } finally {
      inFlightRenders.delete(pageNumber);
    }
  }

  /*****************************************
   * ONMOUNT / ONDESTROY
   *****************************************/
  onMount(async () => {
    // Setup responsive viewport detection
    const mediaQuery = window.matchMedia('(max-width: 768px)');
    isMobileView = mediaQuery.matches;

    const handleResize = () => {
      isMobileView = mediaQuery.matches;
      windowWidth = window.innerWidth;

      // Adjust zoom/scale on window resize if needed
      const newDocumentWidth = canvasContainer?.clientWidth || 0;
      if (documentWidth > 0 && newDocumentWidth !== documentWidth) {
        documentWidth = newDocumentWidth;
        handleScroll(); // Re-evaluate visible pages
      }
    };

    mediaQuery.addEventListener('change', handleResize);
    window.addEventListener('resize', handleResize);

    // Create a ResizeObserver to monitor container size changes
    containerObserver = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const newWidth = entry.contentRect.width;
        if (newWidth !== documentWidth) {
          documentWidth = newWidth;
          handleScroll();
          const needsRerender = Array.from(renderedPages);
          // Update already rendered pages to better fit the new size
          Promise.all(needsRerender.map(pageNum => {
            const pageElement = document.getElementById(`page-${pageNum}`);
            if (pageElement) {
              return updatePageRender(pageNum, pageElement);
            }
            return Promise.resolve();
          }));
        }
      }
    });

    if (!url) {
      error = 'No URL provided for PDF';
      isLoading = false;
      return;
    }

    try {
      const loadingTask = pdfjs.getDocument(url);
      pdfDocument = await loadingTask.promise;
      if (destroyed) return;

      totalPages = pdfDocument.numPages;
      documentWidth = canvasContainer?.clientWidth || 0;

      // Start observing the container for resize events
      if (canvasContainer && containerObserver) {
        containerObserver.observe(canvasContainer);
      }

      // Initial rendering of the first few visible pages
      await attemptInitialRender();

    } catch (e) {
      console.error('Error loading PDF:', e);
      error = `Failed to load PDF: ${e.message}`;
    } finally {
      isLoading = false;
    }

    return () => {
      mediaQuery.removeEventListener('change', handleResize);
      window.removeEventListener('resize', handleResize);
      if (containerObserver) {
        containerObserver.disconnect();
      }
    };
  });

  onDestroy(() => {
    destroyed = true;
    if (canvasContainer) {
      canvasContainer.removeEventListener('scroll', handleScroll);
    }
    if (containerObserver) {
      containerObserver.disconnect();
    }
  });

  /*****************************************
   * INITIAL RENDER
   *****************************************/
  let retryCount = 0;
  const MAX_RETRIES = 5;

  async function attemptInitialRender() {
    if (!canvasContainer) {
      if (retryCount < MAX_RETRIES) {
        retryCount++;
        setTimeout(attemptInitialRender, 100);
      } else {
        console.error('Max retries reached. Canvas container not found.');
        error = 'Failed to initialize PDF viewer';
        isLoading = false;
      }
      return;
    }

    // Pre-load first few pages
    const initialCount = Math.min(3, totalPages);
    for (let i = 1; i <= initialCount; i++) {
      await renderPage(i);
    }

    calculatePagePositions();

    // Start on page 1
    if (!scrollToPage) {
      currentPage = 1;
    }

    // Listen for scrolling
    canvasContainer.addEventListener('scroll', handleScroll);

    // Trigger handleScroll once to render more if needed
    handleScroll();
  }

  /*****************************************
   * PAGE NAVIGATION
   *****************************************/
  async function scrollTo(pageNumber: number) {
    if (!pdfDocument || !canvasContainer) return;
    if (pageNumber < 1 || pageNumber > totalPages) return;

    // Render target page (and neighbors) so it exists in the DOM
    const neighbors = [pageNumber - 1, pageNumber, pageNumber + 1].filter(
      p => p >= 1 && p <= totalPages
    );
    await Promise.all(neighbors.map(p => renderPage(p)));

    calculatePagePositions();

    const offset = pagePositions[pageNumber - 1] || 0;
    canvasContainer.scrollTo({ top: offset, behavior: 'smooth' });

    currentPage = pageNumber;
    dispatch('pagechange', { page: currentPage });
  }

  function nextPage() {
    if (canGoToNextPage) {
      scrollTo(currentPage + 1);
    }
  }

  function previousPage() {
    if (canGoToPrevPage) {
      scrollTo(currentPage - 1);
    }
  }

  function handleManualPageInput() {
    const pageNumber = parseInt(manualPageInput, 10);
    if (!isNaN(pageNumber) && pageNumber > 0 && pageNumber <= totalPages) {
      scrollTo(pageNumber);
    }
    manualPageInput = '';
  }

  /*****************************************
   * ZOOM
   *****************************************/
  function zoomIn() {
    setZoom(currentScale + 0.1);
  }

  function zoomOut() {
    setZoom(currentScale - 0.1);
  }

  function resetZoom() {
    setZoom(1);
  }

  function setZoom(newScale: number) {
    // Limit scale between reasonable bounds
    newScale = Math.max(0.5, Math.min(3.0, newScale));
    if (newScale === currentScale) return;

    currentScale = newScale;
    zoomPercentage = Math.round(currentScale * 100);

    // Re-render visible pages with new scale
    reRenderVisiblePages();
  }

  async function reRenderVisiblePages() {
    if (!canvasContainer) return;

    const { first, last } = getVisiblePages();

    const buffer = 2;
    const renderFrom = Math.max(1, first - buffer);
    const renderTo = Math.min(totalPages, last + buffer);

    // Update scales for already rendered pages
    const existingPages = canvasContainer.querySelectorAll('.page-wrapper');
    existingPages.forEach(async pageEl => {
      const pageNum = parseInt(pageEl.id.split('-')[1]);
      if (pageNum >= renderFrom && pageNum <= renderTo) {
        await updatePageRender(pageNum, pageEl as HTMLElement);
      } else {
        // Remove pages outside the visible range to save memory
        pageEl.remove();
        renderedPages.delete(pageNum);
      }
    });

    // Render pages that haven't been rendered yet
    for (let p = renderFrom; p <= renderTo; p++) {
      if (!document.getElementById(`page-${p}`)) {
        await renderPage(p);
      }
    }

    calculatePagePositions();

    const newPage = getCurrentPage();
    if (newPage !== currentPage) {
      currentPage = newPage;
      dispatch('pagechange', { page: currentPage });
    }
  }

  /*****************************************
   * SEARCH & HIGHLIGHT
   *****************************************/
  async function scrollToResult(result: SearchResult) {
    await scrollTo(result.page);
    await new Promise(resolve => setTimeout(resolve, 200)); // short delay for render
    highlightSearchResult(result);
  }

  function highlightSearchResult(result: SearchResult) {
    // Clear previous highlights
    clearHighlights();

    const pageElement = document.getElementById(`page-${result.page}`);
    if (!pageElement) return;

    const textLayerDiv = pageElement.querySelector('.text-layer') as HTMLDivElement;
    if (!textLayerDiv) return;

    // Find all spans that contain the search text
    const spans = Array.from(textLayerDiv.querySelectorAll('span'));
    let resultSpans = spans.filter(span =>
      span.textContent && span.textContent.toLowerCase().includes(result.text.toLowerCase())
    );

    if (resultSpans.length > 0) {
      // Highlight matching spans
      resultSpans.forEach(span => {
        const text = span.textContent || '';
        const searchRegex = new RegExp(`(${result.text})`, 'gi');
        span.innerHTML = text.replace(searchRegex, '<span class="enhanced-highlight">$1</span>');
        span.classList.add('highlight-search');
      });

      // Scroll the first match into view
      resultSpans[0].scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
  }

  function highlightElementById(elementId: string) {
    const parts = elementId.split('-');
    if (parts.length !== 2) return;

    const pageNumber = parseInt(parts[0]);
    if (isNaN(pageNumber) || pageNumber < 1 || pageNumber > totalPages) return;

    // First scroll to the page
    scrollTo(pageNumber).then(() => {
      // Wait for rendering to complete
      setTimeout(() => {
        const pageElement = document.getElementById(`page-${pageNumber}`);
        if (!pageElement) return;

        const element = pageElement.querySelector(`[data-element-id="${elementId}"]`);
        if (element) {
          // Clear previous highlights
          clearHighlights();

          // Highlight this element
          element.classList.add('highlight-search');
          element.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
      }, 300);
    });
  }

  function toggleSearchBar() {
    if (isSearchMinimized) {
      isSearchMinimized = false;
      isSearchBarVisible = true;
      if (searchResults.length > 0) {
        isResultListVisible = true;
      }
    } else {
      isSearchBarVisible = !isSearchBarVisible;
      if (!isSearchBarVisible) {
        isResultListVisible = false;
      } else if (searchResults.length > 0) {
        isResultListVisible = true;
      }
    }
  }

  function minimizeSearch() {
    isSearchMinimized = true;
    isResultListVisible = false;
  }

  function handleSearchKeydown(event: KeyboardEvent) {
    if (event.key === 'Enter') {
      performSearch();
    }
  }

  function navigateToNextSearchResult() {
    if (searchResults.length === 0) return;

    currentSearchIndex = (currentSearchIndex + 1) % searchResults.length;
    scrollToResult(searchResults[currentSearchIndex]);
  }

  function navigateToPrevSearchResult() {
    if (searchResults.length === 0) return;

    currentSearchIndex = (currentSearchIndex - 1 + searchResults.length) % searchResults.length;
    scrollToResult(searchResults[currentSearchIndex]);
  }

  // Make these functions available for potential external calls
  export function searchAndNavigate(pageNumber: number, elementId: string) {
    highlightElementById(`${pageNumber}-${elementId}`);
  }

  // Export this function for external use
  export { performSearch };
</script>

<div class="pdf-viewer">
  <div class="toolbar">
    <div class="toolbar-group">
      <button
        on:click={previousPage}
        disabled={!canGoToPrevPage}
        class="tool-button {!canGoToPrevPage ? 'disabled' : ''}"
        title="Previous page"
      >
        <Icon name="chevron_left" size="18px" />
      </button>

      <div class="page-input-group">
        <input
          type="text"
          placeholder="{currentPage}"
          bind:value={manualPageInput}
          on:keydown={(e) => e.key === 'Enter' && handleManualPageInput()}
          on:blur={handleManualPageInput}
          class="page-input"
          title="Go to page"
        />
        <span class="page-separator">/ {totalPages}</span>
      </div>

      <button
        on:click={nextPage}
        disabled={!canGoToNextPage}
        class="tool-button {!canGoToNextPage ? 'disabled' : ''}"
        title="Next page"
      >
        <Icon name="chevron_right" size="18px" />
      </button>
    </div>

    <div class="toolbar-group">
      <button
        on:click={zoomOut}
        class="tool-button"
        title="Zoom out"
      >
        <Icon name="remove" size="18px" />
      </button>

      <span class="zoom-indicator">{zoomPercentage}%</span>

      <button
        on:click={zoomIn}
        class="tool-button"
        title="Zoom in"
      >
        <Icon name="add" size="18px" />
      </button>

      <button
        on:click={resetZoom}
        class="reset-button"
        title="Reset zoom"
      >
        100%
      </button>
    </div>

    <div class="toolbar-group">
      <button
        on:click={toggleSearchBar}
        class="tool-button {isSearchBarVisible ? 'active' : ''}"
        title="Search document"
      >
        <Icon name="search" size="18px" />
      </button>
    </div>
  </div>

  <!-- Search UI -->
  {#if isSearchBarVisible}
    <div class="search-panel {isSearchMinimized ? 'minimized' : ''}">
      <div class="search-header">
        <span class="search-title">Search Document</span>
        <div class="search-controls">
          <button
            on:click={minimizeSearch}
            class="icon-button"
            title="Minimize search"
          >
            <Icon name="expand_less" size="18px" />
          </button>
          <button
            on:click={clearSearch}
            class="icon-button"
            title="Close search"
          >
            <Icon name="close" size="18px" />
          </button>
        </div>
      </div>

      <div class="search-input-group">
        <input
          type="text"
          placeholder="Search document..."
          bind:value={searchQuery}
          on:input={debouncedSearch}
          on:keydown={handleSearchKeydown}
          class="search-input"
        />
        <button
          on:click={performSearch}
          class="search-button"
        >
          <Icon name="search" size="18px" />
        </button>
      </div>

      {#if searchResults.length > 0}
        <div class="search-navigation">
          <span class="results-count">{searchResults.length} results found</span>
          <div class="navigation-controls">
            <button
              on:click={navigateToPrevSearchResult}
              class="nav-button"
              disabled={searchResults.length <= 1}
              title="Previous result"
            >
              <Icon name="arrow_upward" size="16px" />
            </button>
            <span class="result-counter">
              {currentSearchIndex + 1} / {searchResults.length}
            </span>
            <button
              on:click={navigateToNextSearchResult}
              class="nav-button"
              disabled={searchResults.length <= 1}
              title="Next result"
            >
              <Icon name="arrow_downward" size="16px" />
            </button>
          </div>
        </div>

        {#if isResultListVisible}
          <div class="search-results">
            {#each displayedResults as result, index}
              <div
                class="result-item {index === currentSearchIndex % resultsPerPage ? 'active' : ''}"
                on:click={() => {
                  currentSearchIndex = (currentResultsPage - 1) * resultsPerPage + index;
                  scrollToResult(result);
                }}
                on:keydown={(e) => {
                  if (e.key === 'Enter') {
                    currentSearchIndex = (currentResultsPage - 1) * resultsPerPage + index;
                    scrollToResult(result);
                  }
                }}
                tabindex="0"
                role="button"
              >
                <div class="result-header">
                  <span class="result-page">Page {result.page}</span>
                  <span class="result-number">Result {(currentResultsPage - 1) * resultsPerPage + index + 1}</span>
                </div>
                <div class="result-preview">
                  {@html result.snippet}
                </div>
              </div>
            {/each}

            {#if searchResults.length > resultsPerPage}
              <div class="pagination">
                <button
                  on:click={prevResultsPage}
                  disabled={currentResultsPage === 1}
                  class="pagination-button {currentResultsPage === 1 ? 'disabled' : ''}"
                >
                  Previous
                </button>
                <span class="pagination-info">
                  Page {currentResultsPage} of {Math.ceil(searchResults.length / resultsPerPage)}
                </span>
                <button
                  on:click={nextResultsPage}
                  disabled={currentResultsPage * resultsPerPage >= searchResults.length}
                  class="pagination-button {currentResultsPage * resultsPerPage >= searchResults.length ? 'disabled' : ''}"
                >
                  Next
                </button>
              </div>
            {/if}
          </div>
        {/if}
      {/if}
    </div>
  {/if}

  <!-- Searching overlay -->
  {#if isSearching}
    <div class="search-overlay">
      <div class="spinner"></div>
      <span>Searching... ({searchResults.length} found)</span>
      <button on:click={() => searchCancelled = true} class="cancel-button">
        Cancel
      </button>
    </div>
  {/if}

  <!-- Main PDF container -->
  <div class="pdf-content">
    {#if error}
      <div class="error-container">
        <div class="error-message">
          <Icon name="error" size="48px" class="error-icon" />
          <p>{error}</p>
        </div>
      </div>
    {:else if isLoading}
      <div class="loading-overlay">
        <div class="spinner large"></div>
        <span class="loading-text">Loading PDF...</span>
      </div>
    {:else}
      <!-- Scrollable container for PDF pages with improved styling -->
      <div
        bind:this={canvasContainer}
        class="pdf-container"
      >
        <!-- Pages will be rendered here by JavaScript -->
      </div>
    {/if}
  </div>
</div>

<style>
  /* Modern PDF viewer styling */
  .pdf-viewer {
    position: relative;
    width: 100%;
    height: 100%;
    display: flex;
    flex-direction: column;
    background-color: #f8fafc;
    --pdf-primary: #3b82f6;
    --pdf-secondary: #1e40af;
    --pdf-text: #1f2937;
    --pdf-background: #f1f5f9;
    --pdf-border: #e5e7eb;
    --pdf-highlight: #fef08a;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
  }

  /* Toolbar styling */
  .toolbar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 16px;
    border-bottom: 1px solid var(--pdf-border);
    background-color: white;
    min-height: 56px;
    position: sticky;
    top: 0;
    z-index: 10;
  }

  .toolbar-group {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .tool-button {
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 6px;
    background-color: transparent;
    border: none;
    color: #64748b;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .tool-button:hover {
    background-color: #f1f5f9;
    color: #334155;
  }

  .tool-button.active {
    background-color: #e0f2fe;
    color: var(--pdf-primary);
  }

  .tool-button.disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .page-input-group {
    display: flex;
    align-items: center;
    gap: 4px;
    background-color: #f1f5f9;
    padding: 4px 8px;
    border-radius: 6px;
  }

  .page-input {
    width: 40px;
    padding: 4px 6px;
    border-radius: 4px;
    border: 1px solid #e2e8f0;
    text-align: center;
    font-size: 13px;
  }

  .page-input:focus {
    outline: none;
    border-color: var(--pdf-primary);
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1);
  }

  .page-separator {
    font-size: 13px;
    color: #64748b;
  }

  .zoom-indicator {
    font-size: 13px;
    color: #64748b;
    min-width: 42px;
    text-align: center;
  }

  .reset-button {
    padding: 4px 8px;
    border-radius: 4px;
    border: none;
    background-color: #f1f5f9;
    color: #64748b;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .reset-button:hover {
    background-color: #e2e8f0;
  }

  /* Search panel styling */
  .search-panel {
    position: absolute;
    top: 65px;
    left: 50%;
    transform: translateX(-50%);
    width: 380px;
    max-width: 90%;
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    padding: 16px;
    z-index: 20;
    transition: all 0.3s ease;
  }

  .search-panel.minimized {
    height: 0;
    padding: 0;
    overflow: hidden;
  }

  .search-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }

  .search-title {
    font-size: 14px;
    font-weight: 600;
    color: var(--pdf-text);
  }

  .search-controls {
    display: flex;
    gap: 4px;
  }

  .icon-button {
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 4px;
    background-color: transparent;
    border: none;
    color: #64748b;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .icon-button:hover {
    background-color: #f1f5f9;
    color: #334155;
  }

  .search-input-group {
    position: relative;
    margin-bottom: 12px;
  }

  .search-input {
    width: 100%;
    padding: 8px 40px 8px 12px;
    border-radius: 6px;
    border: 1px solid #e2e8f0;
    font-size: 13px;
    transition: all 0.2s ease;
  }

  .search-input:focus {
    outline: none;
    border-color: var(--pdf-primary);
    box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1);
  }

  .search-button {
    position: absolute;
    right: 4px;
    top: 50%;
    transform: translateY(-50%);
    width: 32px;
    height: 32px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 4px;
    border: none;
    background-color: var(--pdf-primary);
    color: white;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .search-button:hover {
    background-color: var(--pdf-secondary);
  }

  .search-navigation {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
  }

  .results-count {
    font-size: 12px;
    color: #64748b;
  }

  .navigation-controls {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  .nav-button {
    width: 24px;
    height: 24px;
    display: flex;
    align-items: center;
    justify-content: center;
    border-radius: 4px;
    border: none;
    background-color: #f1f5f9;
    color: var(--pdf-primary);
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .nav-button:hover:not(:disabled) {
    background-color: #e0f2fe;
  }

  .nav-button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .result-counter {
    font-size: 12px;
    color: #64748b;
  }

  .search-results {
    max-height: 300px;
    overflow-y: auto;
    border-radius: 6px;
    border: 1px solid #e2e8f0;
  }

  .result-item {
    padding: 10px 12px;
    cursor: pointer;
    border-bottom: 1px solid #e2e8f0;
    transition: all 0.2s ease;
  }

  .result-item:last-child {
    border-bottom: none;
  }

  .result-item:hover {
    background-color: #f8fafc;
  }

  .result-item.active {
    background-color: #e0f2fe;
  }

  .result-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 4px;
  }

  .result-page {
    font-size: 12px;
    font-weight: 600;
    color: var(--pdf-primary);
  }

  .result-number {
    font-size: 11px;
    color: #94a3b8;
  }

  .result-preview {
    font-size: 12px;
    color: #64748b;
    word-break: break-word;
  }

  .result-preview :global(mark) {
    background-color: var(--pdf-highlight);
    color: var(--pdf-text);
    padding: 0 2px;
    border-radius: 2px;
  }

  .pagination {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    background-color: #f8fafc;
    border-top: 1px solid #e2e8f0;
  }

  .pagination-button {
    padding: 4px 8px;
    border-radius: 4px;
    border: none;
    background-color: #e2e8f0;
    color: #64748b;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.2s ease;
  }

  .pagination-button:hover:not(:disabled) {
    background-color: #cbd5e1;
  }

  .pagination-button.disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .pagination-info {
    font-size: 12px;
    color: #64748b;
  }

  /* Search overlay */
  .search-overlay {
    position: absolute;
    top: 65px;
    left: 50%;
    transform: translateX(-50%);
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 10px 16px;
    background-color: white;
    border-radius: 8px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    z-index: 30;
  }

  .spinner {
    width: 16px;
    height: 16px;
    border: 2px solid #e2e8f0;
    border-top-color: var(--pdf-primary);
    border-radius: 50%;
    animation: spin 1s linear infinite;
  }

  .spinner.large {
    width: 32px;
    height: 32px;
    border-width: 3px;
  }

  @keyframes spin {
    to { transform: rotate(360deg); }
  }

  .cancel-button {
    padding: 4px 8px;
    border-radius: 4px;
    border: none;
    background-color: #fee2e2;
    color: #dc2626;
    font-size: 12px;
    cursor: pointer;
    transition: all 0.2s ease;
    margin-left: 8px;
  }

  .cancel-button:hover {
    background-color: #fecaca;
  }

  /* PDF content area */
  .pdf-content {
    flex: 1;
    position: relative;
    overflow: hidden;
  }

  /* PDF container */
  .pdf-container {
    width: 100%;
    height: 100%;
    padding: 20px;
    overflow-y: auto;
    background-color: var(--pdf-background);
    display: flex;
    flex-direction: column;
    align-items: center;
  }

  /* Error container */
  .error-container {
    width: 100%;
    height: 100%;
    display: flex;
    align-items: center;
    justify-content: center;
    background-color: #fee2e2;
    padding: 20px;
  }

  .error-message {
    max-width: 600px;
    text-align: center;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 12px;
  }

  .error-icon {
    color: #dc2626;
  }

  .error-message p {
    color: #b91c1c;
    font-size: 16px;
    margin: 0;
  }

  /* Loading overlay */
  .loading-overlay {
    position: absolute;
    inset: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    background-color: white;
    background-opacity: 0.9;
    gap: 16px;
    z-index: 10;
  }

  .loading-text {
    color: var(--pdf-primary);
    font-size: 16px;
    font-weight: 500;
  }

  /* Make page styles available globally */
  :global(.page-wrapper) {
    margin-bottom: 20px;
    width: 100%;
    display: flex;
    justify-content: center;
  }

  :global(.page-container) {
    position: relative;
    background-color: white;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    border-radius: 4px;
    overflow: hidden;
  }

  /* Text layer styling */
  :global(.text-layer) {
    position: absolute;
    left: 0;
    top: 0;
    right: 0;
    bottom: 0;
    overflow: hidden;
    opacity: 1;
    line-height: normal;
    text-align: initial;
    user-select: text;
  }

  :global(.text-element) {
    color: transparent;
    position: absolute;
    white-space: pre;
    cursor: text;
    transform-origin: 0% 0%;
    pointer-events: auto;
  }

  /* Highlight styles */
  :global(.enhanced-highlight) {
    background-color: var(--pdf-highlight);
    color: black !important;
    border-radius: 2px;
    opacity: 1 !important;
  }

  :global(.highlight-search) {
    background-color: var(--pdf-highlight);
    box-shadow: 0 0 0 2px #fde047;
    transition: background-color 0.3s ease;
    border-radius: 2px;
  }

  /* Custom scrollbar */
  .pdf-container, .search-results {
    scrollbar-width: thin;
    scrollbar-color: #cbd5e1 transparent;
  }

  .pdf-container::-webkit-scrollbar, .search-results::-webkit-scrollbar {
    width: 6px;
  }

  .pdf-container::-webkit-scrollbar-track, .search-results::-webkit-scrollbar-track {
    background: transparent;
  }

  .pdf-container::-webkit-scrollbar-thumb, .search-results::-webkit-scrollbar-thumb {
    background-color: #cbd5e1;
    border-radius: 3px;
  }

  /* Responsive styles */
  @media (max-width: 768px) {
    .toolbar {
      padding: 8px;
    }

    .toolbar-group {
      gap: 4px;
    }

    .search-panel {
      width: 90%;
    }

    .pdf-container {
      padding: 10px;
    }
  }
</style>
