/***************************************************
 * pdfViewerEnhancements.js
 *
 * Core helper functions for PDF viewer enhancements.
 *
 * - Zoom management (setupZoom)
 * - Text layer styling (enhanceTextLayer)
 * - Text selection improvements (enhanceTextSelection)
 * - Search term highlighting (improveSearchHighlighting)
 * - Simple element highlight (highlightElement)
 * - (Optional) Element search on a page (searchElementOnPage)
 ***************************************************/

// 1) Zoom Management
export function setupZoom(initialScale) {
  let currentScale = initialScale;
  const MAX_SCALE = 3;
  const MIN_SCALE = 0.5;

  const zoomIn = () => setZoom(currentScale + 0.1);
  const zoomOut = () => setZoom(currentScale - 0.1);
  const resetZoom = () => setZoom(1);

  function setZoom(newScale) {
    newScale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, newScale));
    currentScale = newScale;
    return currentScale;
  }

  return {
    zoomIn,
    zoomOut,
    resetZoom,
    setZoom,
    getCurrentScale: () => currentScale
  };
}

// 2) Basic styling for the text layer div, typically used after creating it
export function enhanceTextLayer(textLayer, viewport) {
  // Position and size
  textLayer.style.left = '0';
  textLayer.style.top = '0';
  textLayer.style.right = '0';
  textLayer.style.bottom = '0';
  textLayer.style.position = 'absolute';

  // Store the rotation in a data attribute, if needed for debugging
  if (viewport && typeof viewport.rotation === 'number') {
    textLayer.setAttribute('data-main-rotation', viewport.rotation.toString());
  }
}

// 3) Make text selectable & avoid weird double-click behaviors
export function enhanceTextSelection(textLayer) {
  textLayer.style.userSelect = 'text';
  textLayer.style.cursor = 'text';

  textLayer.addEventListener('mousedown', (e) => {
    if (e.detail > 1) {
      // Prevent double/triple-click from automatically selecting large sections
      e.preventDefault();
    }
  });
}

// 4) Highlight occurrences of a given search term within the text layer
export function improveSearchHighlighting(textLayer, searchTerm) {
  if (!searchTerm) return;

  const searchRegex = new RegExp(searchTerm, 'gi');
  const textNodes = Array.from(textLayer.childNodes);

  // Remove any existing highlights
  const existingHighlights = textLayer.querySelectorAll('.enhanced-highlight');
  existingHighlights.forEach(el => {
    // Revert highlight <span> back to plain text
    el.outerHTML = el.innerHTML;
  });

  // Wrap matches in <span class="enhanced-highlight">
  textNodes.forEach(node => {
    if (node.nodeType === Node.TEXT_NODE) {
      const text = node.textContent || '';
      const matches = text.match(searchRegex);

      if (matches) {
        const fragment = document.createDocumentFragment();
        let lastIndex = 0;

        matches.forEach(match => {
          const index = text.indexOf(match, lastIndex);
          if (index > lastIndex) {
            // Append text up to the match
            fragment.appendChild(
              document.createTextNode(text.slice(lastIndex, index))
            );
          }

          // Create highlight span
          const span = document.createElement('span');
          span.className = 'enhanced-highlight';
          span.textContent = match;
          fragment.appendChild(span);

          lastIndex = index + match.length;
        });

        // Append remaining text
        if (lastIndex < text.length) {
          fragment.appendChild(
            document.createTextNode(text.slice(lastIndex))
          );
        }

        // Replace original node with the new fragment
        node.parentNode.replaceChild(fragment, node);
      }
    }
  });
}

// 5) Temporarily highlight an element by adding a CSS class, then remove it.
export function highlightElement(element) {
  if (!element) return;
  element.classList.add('highlight-search');
  setTimeout(() => {
    element.classList.remove('highlight-search');
  }, 3000);
}

// 6) (Optional) Find an element by [data-element-id] on a given page
export function searchElementOnPage(page, elementId) {
  const element = page.querySelector(`[data-element-id="${elementId}"]`);
  if (element) {
    return {
      found: true,
      position: element.getBoundingClientRect()
    };
  }
  return { found: false };
}

/******************************************************
 * NOTE: Functions that would cause concurrency or
 * canvas usage changes (like `renderPage`) live in
 * your PdfViewer.svelte. This helper file is purely
 * for text layering, zoom, and highlighting logic.
 ******************************************************/
