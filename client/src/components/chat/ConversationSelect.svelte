<!-- ConversationSelect.svelte -->
<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import { setActiveConversationId } from '$s/chat';
  import Icon from '$c/Icon.svelte';

  export let conversations = [];
  export let activeId = null;

  const dispatch = createEventDispatcher();

  function selectConversation(id) {
    setActiveConversationId(id);
    dispatch('select', { id });
  }

  $: activeConversation = conversations.find(c => c.id === activeId) || (conversations.length > 0 ? conversations[0] : null);
  $: activeTitle = activeConversation?.title || 'Select conversation';

  let isDropdownOpen = false;

  function toggleDropdown() {
    isDropdownOpen = !isDropdownOpen;
  }

  function handleBlur() {
    // Add a small delay to allow for click events to register first
    setTimeout(() => {
      isDropdownOpen = false;
    }, 100);
  }
</script>

<div class="relative">
  <button
    class="px-3 py-2 bg-white border rounded-md text-sm flex items-center justify-between min-w-[12rem] shadow-sm hover:bg-gray-50"
    on:click={toggleDropdown}
    on:blur={handleBlur}
  >
    <span class="truncate">{activeTitle}</span>
    <Icon name={isDropdownOpen ? "expand_less" : "expand_more"} size="18px" class="ml-1" />
  </button>

  {#if isDropdownOpen && conversations.length > 0}
    <div class="absolute top-full left-0 z-10 mt-1 w-full bg-white border rounded-md shadow-lg max-h-60 overflow-y-auto">
      {#each conversations as conversation}
        <button
          class="w-full text-left px-3 py-2 hover:bg-gray-100 text-sm truncate {conversation.id === activeId ? 'bg-blue-50 text-blue-700' : ''}"
          on:click={() => selectConversation(conversation.id)}
        >
          {conversation.title || `Conversation ${conversation.id}`}
        </button>
      {/each}
    </div>
  {/if}
</div>
