<!-- PDFViewerWrapper.svelte -->
<script lang="ts">
    import { onMount, onDestroy, tick } from 'svelte';
    import PdfViewer from './PdfViewer.svelte';
    import ExpandableChatPanel from './chat/ExpandableChatPanel.svelte';
    import DocumentSidebar from './DocumentSidebar.svelte';
    import { generatePdfViewerUrl } from './pdfUtils';
    import { sendMessage } from '$s/chat/index';
    import Icon from '$c/Icon.svelte';

    export let pdfId: string;

    let currentPage = 1;
    let scrollToElementId: string | null = null;
    let pdfViewerComponent: PdfViewer;
    let chatPanelWidth = '33%';
    let isSidebarVisible = true; // Ensure this is true by default
    let documentName = "Document";
    let isMobileView = false;
    let activeTab = 'outline';
    let isResizing = false;
    let startX = 0;
    let startWidth = 0;

    // PDF Tabs functionality with debug tracking
    let pdfTabs = [{ id: pdfId, name: documentName, isPrimary: true }];
    let activeTabIndex = 0;
    let tabsUpdated = 0; // Counter to force reactivity updates

    // Reactive declarations with debug logs
    $: {
        console.log("Tabs updated:", pdfTabs, "activeTabIndex:", activeTabIndex);
        tabsUpdated++;
    }
    $: currentPdfId = pdfTabs[activeTabIndex]?.id || pdfId;
    $: currentDocumentName = pdfTabs[activeTabIndex]?.name || documentName;
    $: pdfUrl = generatePdfViewerUrl(currentPdfId);
    $: console.log("Current PDF settings:", {
        currentPdfId,
        currentDocumentName,
        pdfUrl,
        tabsCount: pdfTabs.length,
        activeTabIndex,
        showTabs: pdfTabs.length > 1
    });

    onMount(async () => {
        console.log("PDFViewerWrapper mounted");

        const mediaQuery = window.matchMedia('(max-width: 768px)');
        isMobileView = mediaQuery.matches;

        const handleResize = () => {
            isMobileView = mediaQuery.matches;
            if (isMobileView) {
                chatPanelWidth = '100%';
            } else {
                chatPanelWidth = '33%';
            }
        };

        mediaQuery.addEventListener('change', handleResize);

        // Explicitly force sidebar visible for desktop on initial load
        if (!isMobileView) {
            isSidebarVisible = true;
            console.log("Setting sidebar visible on desktop");
        }

        // Fetch document name if available
        try {
            const name = await fetchDocumentName(pdfId);
            documentName = name;

            // Update in tabs - create a new array to ensure reactivity
            pdfTabs = pdfTabs.map(tab =>
                tab.id === pdfId ? {...tab, name} : tab
            );

            console.log("Updated primary document name:", documentName);
        } catch (e) {
            console.error("Error fetching primary document name:", e);
        }

        // Wait a tick to ensure all components are initialized
        await tick();
        console.log("Initial PDFViewerWrapper setup complete");

        return () => {
            mediaQuery.removeEventListener('change', handleResize);
        };
    });

    async function fetchDocumentName(docId: string): Promise<string> {
        console.log("Fetching document name for:", docId);
        try {
            const res = await fetch(`/api/documents/${docId}`);
            if (!res.ok) {
                throw new Error(`Failed to fetch document name: ${res.status}`);
            }
            const data = await res.json();
            console.log("Fetched document data:", data);
            return data?.name || `Document ${docId}`;
        } catch (e) {
            console.error("Error fetching document name:", e);
            return `Document ${docId}`;
        }
    }

    function handleSubmit(content: string, useStreaming: boolean, useResearch: boolean) {
        sendMessage(content, {
            useStreaming,
            useResearch
        });
    }

    function handlePdfSearch(event) {
        const { pdfId: targetPdfId, pageNumber, elementId } = event.detail;
        console.log("PDF search event:", event.detail);

        // Check if we need to switch tabs or open a new one
        if (targetPdfId !== currentPdfId) {
            console.log(`Need to switch from ${currentPdfId} to ${targetPdfId}`);
            openDocumentInTab(targetPdfId);
        }

        // Once the correct PDF is active, search for the element
        setTimeout(() => {
            if (pdfViewerComponent && pdfViewerComponent.searchAndNavigate) {
                console.log("Searching for element:", elementId, "on page:", pageNumber);
                pdfViewerComponent.searchAndNavigate(pageNumber, elementId);
            } else {
                console.warn("PDF viewer component or searchAndNavigate method not available");
            }
        }, 100); // Short delay to ensure tab switching is complete
    }

    async function openDocumentInTab(docId, docName) {
        console.log("PDFViewerWrapper: Opening document in tab", docId, docName);

        // Check if already open
        const existingTabIndex = pdfTabs.findIndex(tab => tab.id === docId);
        if (existingTabIndex >= 0) {
            console.log("Tab already exists at index:", existingTabIndex, "switching to it");
            switchPdfTab(existingTabIndex);
            return;
        }

        // If no name provided, try to fetch it
        if (!docName) {
            try {
                docName = await fetchDocumentName(docId);
                console.log("Fetched document name:", docName);
            } catch (e) {
                console.error("Error fetching document name:", e);
                docName = `Document ${docId}`;
            }
        }

        console.log("Creating new tab with:", { id: docId, name: docName });

        // Create a temporary tab object for debugging
        const newTab = {
            id: docId,
            name: docName,
            isPrimary: false
        };
        console.log("New tab object:", newTab);

        // CRITICAL: Create a completely new array to ensure Svelte reactivity
        const newTabs = [...pdfTabs, newTab];
        console.log("New tabs array:", newTabs);

        // Update the tabs array - this should trigger reactive updates
        pdfTabs = newTabs;

        console.log("Updated pdfTabs:", pdfTabs);
        console.log("Tabs length:", pdfTabs.length);

        // Force a UI update
        await tick();

        // Switch to the new tab after a delay to ensure DOM is updated
        setTimeout(() => {
            const newIndex = pdfTabs.length - 1;
            console.log("Switching to tab index:", newIndex);
            switchPdfTab(newIndex);
        }, 50);
    }

    function switchPdfTab(index) {
        console.log(`Switching from tab ${activeTabIndex} to ${index}`);

        if (index !== activeTabIndex && pdfTabs[index]) {
            activeTabIndex = index;
            console.log("New active tab:", pdfTabs[activeTabIndex]);

            // Reset to first page when switching tabs
            currentPage = 1;
            scrollToElementId = null;
        }
    }

    function closeTab(index) {
        console.log("Closing tab at index:", index);

        // Prevent closing primary document tab
        if (pdfTabs[index]?.isPrimary) {
            console.log("Cannot close primary document tab");
            return;
        }

        if (index === activeTabIndex) {
            // If closing active tab, switch to primary
            activeTabIndex = 0;
            console.log("Closed active tab, switching to primary");
        } else if (index < activeTabIndex) {
            // If closing tab before active, adjust index
            activeTabIndex--;
            console.log("Closed tab before active, adjusting index to:", activeTabIndex);
        }

        // Remove the tab - create a new array for reactivity
        pdfTabs = pdfTabs.filter((_, i) => i !== index);
        console.log("Tabs after closing:", pdfTabs);
    }

    function handleChatResize(event: CustomEvent<string>) {
        if (!isMobileView) {
            chatPanelWidth = event.detail;
        }
    }

    function toggleSidebar() {
        console.log("Toggling sidebar from", isSidebarVisible, "to", !isSidebarVisible);
        isSidebarVisible = !isSidebarVisible;
    }

    function handleSidebarTabChange(event) {
        activeTab = event.detail.tab;
    }

    function handleSidebarSearch(event) {
        if (pdfViewerComponent && pdfViewerComponent.performSearch) {
            pdfViewerComponent.performSearch(event.detail.query);
        }
    }

    function handleGotoPage(event) {
        currentPage = event.detail.page;
        if (pdfViewerComponent && pdfViewerComponent.scrollTo) {
            pdfViewerComponent.scrollTo(currentPage);
        }
    }

    function handleOpenDocument(event) {
        // CRITICAL: Add extensive logging to debug the event flow
        console.log("PDFViewerWrapper: Received openDocument event", event.detail);

        const { documentId, name } = event.detail;
        console.log(`PDFViewerWrapper: Opening document ${documentId} with name "${name}"`);
        console.log("Current tabs:", JSON.stringify(pdfTabs));

        // Directly modify pdfTabs to force reactivity
        openDocumentInTab(documentId, name);
    }

    // Manual resize handlers
    function startResize(e) {
        isResizing = true;
        startX = e.clientX;
        startWidth = parseInt(chatPanelWidth);

        // Add event listeners for mouse movement and release
        window.addEventListener('mousemove', handleResize);
        window.addEventListener('mouseup', stopResize);

        e.preventDefault(); // Prevent text selection during resize
    }

    function handleResize(e) {
        if (!isResizing) return;

        // Calculate new width as percentage
        const containerWidth = document.body.clientWidth;
        const widthChange = ((e.clientX - startX) / containerWidth) * 100;
        const newWidth = Math.min(Math.max(20, startWidth - widthChange), 60); // Limit between 20% and 60%

        chatPanelWidth = `${newWidth}%`;
        handleChatResize(new CustomEvent('resize', { detail: chatPanelWidth }));
    }

    function stopResize() {
        isResizing = false;
        window.removeEventListener('mousemove', handleResize);
        window.removeEventListener('mouseup', stopResize);
    }
</script>

<div class="app-container flex h-screen overflow-hidden bg-gray-50">
    <!-- Sidebar toggle button - always visible regardless of mode -->
    <button
        class="fixed top-4 left-4 z-50 p-2 bg-white rounded-full shadow-lg hover:bg-gray-100 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500"
        on:click={toggleSidebar}
        aria-label={isSidebarVisible ? "Hide sidebar" : "Show sidebar"}
    >
        <Icon name={isSidebarVisible ? "menu_open" : "menu"} size="24px" />
    </button>

    <!-- Document sidebar with improved visibility -->
    {#if isSidebarVisible}
        <div class="sidebar-container {isMobileView ? 'mobile' : ''}"
             style="width: {isMobileView ? '100%' : '280px'}; z-index: 40;">
            <DocumentSidebar
                pdfId={currentPdfId}
                documentName={currentDocumentName}
                isOutlineVisible={activeTab === 'outline'}
                on:toggleOutline={toggleSidebar}
                on:tabChange={handleSidebarTabChange}
                on:search={handleSidebarSearch}
                on:gotoPage={handleGotoPage}
            />
        </div>
    {/if}

    <!-- PDF Tabs Bar - Always render but conditionally show -->
    <div class="pdf-tabs {isSidebarVisible ? 'with-sidebar' : ''} {pdfTabs.length > 1 ? 'visible' : 'hidden'}"
         style="left: {isSidebarVisible ? '280px' : '0px'}">
        <!-- Debug comment for visibility in DOM inspector -->
        <!-- Tabs count: {pdfTabs.length}, Active tab: {activeTabIndex}, Update counter: {tabsUpdated} -->

        {#each pdfTabs as tab, i (tab.id)}
            <button
                class="pdf-tab {activeTabIndex === i ? 'active' : ''}"
                on:click={() => switchPdfTab(i)}
                title={tab.name}
            >
                <span class="tab-title">{tab.name || `Document ${tab.id}`}</span>
                {#if !tab.isPrimary}
                    <button
                        class="close-tab"
                        on:click|stopPropagation={() => closeTab(i)}
                        title="Close tab"
                    >
                        <Icon name="close" size="14px" />
                    </button>
                {/if}
            </button>
        {/each}
    </div>

    <!-- Main content area with PDF viewer -->
    <div
        class="pdf-viewer-wrapper flex-1 overflow-hidden transition-all {pdfTabs.length > 1 ? 'with-tabs' : ''}"
        style="width: calc(100% - {isSidebarVisible ? '280px' : '0px'} - {isMobileView ? '0px' : chatPanelWidth})"
    >
        <PdfViewer
            bind:this={pdfViewerComponent}
            url={pdfUrl}
            scrollToPage={currentPage.toString()}
            {scrollToElementId}
            on:pagechange={(event) => currentPage = event.detail.page}
        />
    </div>

    <!-- Resize handle (desktop only) -->
    {#if !isMobileView}
        <div
            class="resize-handle"
            on:mousedown={startResize}
            class:is-resizing={isResizing}></div>
    {/if}

    <!-- Chat panel with improved styling -->
    <ExpandableChatPanel
        onSubmit={handleSubmit}
        documentId={parseInt(pdfId)}
        {pdfId}
        onPdfSearch={handlePdfSearch}
        on:resize={handleChatResize}
        on:openDocument={handleOpenDocument}
        class={isMobileView ? "mobile-view" : ""}
    />
</div>

<style>
    .app-container {
        position: relative;
        background-color: #f8fafc;
        overflow: hidden;
    }

    .sidebar-container {
        height: 100%;
        background-color: white;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        z-index: 40;
        flex-shrink: 0;
        border-right: 1px solid #e5e7eb;
    }

    /* Improved PDF tabs styling for better visibility */
    .pdf-tabs {
        position: absolute;
        top: 0;
        right: 0;
        height: 40px;
        display: flex;
        background-color: #f8fafc;
        border-bottom: 1px solid #e5e7eb;
        padding-left: 8px;
        z-index: 25; /* Increased z-index */
        transition: left 0.3s ease;
    }

    .pdf-tabs.with-sidebar {
        left: 280px;
    }

    /* Force visibility when needed */
    .pdf-tabs.visible {
        display: flex !important;
    }

    .pdf-tabs.hidden {
        display: none !important;
    }

    .pdf-tab {
        display: flex;
        align-items: center;
        gap: 8px;
        height: 100%;
        padding: 0 16px;
        background-color: #f1f5f9;
        border: none;
        border-right: 1px solid #e5e7eb;
        color: #64748b;
        font-size: 13px;
        cursor: pointer;
        transition: all 0.2s;
        white-space: nowrap;
        max-width: 200px;
        position: relative;
        /* Add shadow for better visibility */
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
    }

    .pdf-tab.active {
        background-color: white;
        color: #0f172a;
        font-weight: 500;
        /* Make active tab more distinctive */
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        border-bottom: 2px solid #3b82f6;
    }

    .tab-title {
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 150px;
    }

    .close-tab {
        width: 18px;
        height: 18px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 50%;
        border: none;
        background-color: transparent;
        color: #94a3b8;
        padding: 0;
        cursor: pointer;
        transition: all 0.2s;
    }

    .close-tab:hover {
        background-color: #e2e8f0;
        color: #475569;
    }

    .pdf-viewer-wrapper {
        background-color: #f1f5f9;
    }

    /* Make sure there's space for tabs */
    .pdf-viewer-wrapper.with-tabs {
        padding-top: 40px !important; /* Force padding-top */
        margin-top: 0;
    }

    .resize-handle {
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

    .resize-handle.is-resizing {
        background-color: rgba(59, 130, 246, 0.5);
    }

    /* Mobile responsive styles with improvements */
    @media (max-width: 768px) {
        .sidebar-container.mobile {
            position: absolute;
            top: 0;
            left: 0;
            bottom: 0;
            z-index: 40;
            transform: translateX(0);
            box-shadow: 2px 0 15px rgba(0, 0, 0, 0.1);
        }

        .pdf-tabs {
            left: 0 !important;
            width: 100%;
            overflow-x: auto;
            -webkit-overflow-scrolling: touch;
        }

        .pdf-tab {
            flex-shrink: 0;
        }

        .pdf-viewer-wrapper {
            width: 100% !important;
        }

        .mobile-view {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            height: auto;
            max-height: 50%;
            z-index: 40;
            transform: translateY(calc(100% - 40px));
            transition: transform 0.3s ease;
            border-radius: 16px 16px 0 0;
            box-shadow: 0 -4px 20px rgba(0, 0, 0, 0.15);
        }

        .mobile-view.expanded {
            transform: translateY(0);
        }
        .pdf-viewer-container {
          --scale-factor: 1.0;
        }

    }
</style>
