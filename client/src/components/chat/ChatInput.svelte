<!-- ChatInput.svelte -->
<script lang="ts">
	import { createEventDispatcher } from 'svelte';
	import Icon from '$c/Icon.svelte';

	let value = '';
	export let disabled = false;
	export let loading = false;
	export let placeholder = 'Type your message here...';

	const dispatch = createEventDispatcher();

	function handleKeyDown(event: KeyboardEvent) {
		// Check if submission is disabled
		if (disabled || loading) {
			return;
		}

		const isCombo = event.shiftKey || event.ctrlKey || event.altKey || event.metaKey;

		if (event.key !== 'Enter' || isCombo) {
			return;
		}

		if (event.key === 'Enter' && !isCombo && value === '') {
			event.preventDefault();
			return;
		}

		event.preventDefault();
		submitMessage();
	}

	function submitMessage() {
		// Only submit if not disabled or loading
		if (!disabled && !loading && value.trim()) {
			dispatch('submit', value);
			value = '';
		}
	}

	$: height = Math.min(Math.max((value.match(/\n/g)?.length || 0) * 24 + 48, 48), 160);
</script>

<div class="input-container relative">
	<textarea
		class="chat-textarea"
		style:height={height + 'px'}
		bind:value
		on:keydown={handleKeyDown}
		{placeholder}
		disabled={disabled || loading}
		aria-label="Chat message input"
	/>

	<button
		class="send-button {disabled || loading || !value.trim() ? 'disabled' : 'active'}"
		on:click={submitMessage}
		disabled={disabled || loading || !value.trim()}
		title="Send message"
		aria-label="Send message"
	>
		{#if loading}
			<div class="spinner"></div>
		{:else}
			<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
				<path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z" />
			</svg>
		{/if}
	</button>
</div>

<style>
	.input-container {
		width: 100%;
	}

	.chat-textarea {
		width: 100%;
		resize: none;
		border: 1px solid #e2e8f0;
		border-radius: 10px;
		padding: 12px 46px 12px 16px;
		min-height: 48px;
		max-height: 160px;
		font-size: 14px;
		line-height: 1.5;
		background-color: #f8fafc;
		transition: all 0.2s ease;
		box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);
	}

	.chat-textarea:focus {
		outline: none;
		border-color: #3b82f6;
		box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
		background-color: white;
	}

	.chat-textarea:disabled {
		opacity: 0.7;
		cursor: not-allowed;
		background-color: #f1f5f9;
	}

	.send-button {
		position: absolute;
		right: 10px;
		bottom: 10px;
		width: 32px;
		height: 32px;
		border-radius: 50%;
		display: flex;
		align-items: center;
		justify-content: center;
		border: none;
		transition: all 0.2s ease;
		cursor: pointer;
	}

	.send-button.active {
		background-color: #3b82f6;
		color: white;
		box-shadow: 0 2px 4px rgba(59, 130, 246, 0.3);
	}

	.send-button.active:hover {
		background-color: #2563eb;
		transform: scale(1.05);
	}

	.send-button.disabled {
		background-color: #e2e8f0;
		color: #94a3b8;
		cursor: not-allowed;
	}

	.spinner {
		width: 16px;
		height: 16px;
		border: 2px solid rgba(255, 255, 255, 0.3);
		border-radius: 50%;
		border-top-color: white;
		animation: spin 1s linear infinite;
	}

	@keyframes spin {
		to {
			transform: rotate(360deg);
		}
	}
</style>
