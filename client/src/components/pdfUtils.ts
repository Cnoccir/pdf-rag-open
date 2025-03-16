// pdfUtils.ts

export function generatePdfViewerUrl(pdfId: string, page: number, elementId: string): string {
    return `/pdf-viewer/${pdfId}?page=${page}&element_id=${elementId}`;
}

export function scrollToPageAndElement(pageNumber: number, elementId: string | null): void {
    const pageElement = document.getElementById(`page-${pageNumber}`);
    if (pageElement) {
        pageElement.scrollIntoView({ behavior: 'smooth' });

        if (elementId) {
            setTimeout(() => {
                const element = document.getElementById(elementId);
                if (element) {
                    element.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    element.classList.add('highlight-element');
                }
            }, 500);  // Give some time for the page to render
        }
    }
}
