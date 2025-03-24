<script lang="ts">
	import { beforeNavigate } from '$app/navigation';
	import { resetAll, safeSendMessage } from '$s/chat/index';
	import ChatPanel from '$c/chat/ChatPanel.svelte';
	import { onMount } from 'svelte';

	beforeNavigate(resetAll);

	// Use our safer message sending implementation
	function handleSubmit(content: string, useStreaming: boolean, useResearch: boolean) {
		safeSendMessage(content, {
			useStreaming,
			useResearch,
		});
	}

	// Add error handling for component initialization
	onMount(() => {
		console.log("Chat page mounted successfully");
	});
</script>

<div class="chat-panel-wrapper">
	<ChatPanel onSubmit={handleSubmit} />
</div>

<style>
	.chat-panel-wrapper {
		height: 80%;
	}
</style>
