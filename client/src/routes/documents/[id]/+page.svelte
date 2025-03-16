<script lang="ts">
	import type { PageData } from './$types';
	import { beforeNavigate } from '$app/navigation';
	import { resetAll, sendMessage } from '$s/chat/index';
	import PdfViewer from '$c/PdfViewer.svelte';
	import ExpandableChatPanel from '$c/chat/ExpandableChatPanel.svelte';

	export let data: PageData;
	const document = data.document;
	const documentUrl = data.documentUrl;

	let chatPanelWidth = '33%';

	function handleSubmit(content: string, useStreaming: boolean) {
		sendMessage({ role: 'user', content }, { useStreaming, documentId: document.id });
	}

	function handleChatPanelResize(event: CustomEvent<string>) {
		chatPanelWidth = event.detail;
	}

	beforeNavigate(resetAll);
</script>

{#if data.error}
	<div class="error-message">{data.error}</div>
{/if}

{#if document}
	<div class="flex" style="height: calc(100vh - 80px);">
		<div class="pdf-container" style="width: calc(100% - {chatPanelWidth})">
			<PdfViewer url={documentUrl} />
		</div>
		<ExpandableChatPanel
			documentId={document.id}
			onSubmit={handleSubmit}
			on:resize={handleChatPanelResize}
		/>
	</div>
{/if}

<style>
	.error-message {
		color: red;
		padding: 1rem;
	}

	.flex {
		display: flex;
	}

	.pdf-container {
		transition: width 0.3s ease;
	}
</style>
