<script lang="ts">
  import AuthGuard from '$c/AuthGuard.svelte';
  import HealthDashboard from '$c/HealthDashboard.svelte';
  import ErrorModal from '$c/ErrorModal.svelte';
</script>

<AuthGuard />

<Navbar />

<div class="container mx-auto py-6 px-4">
  <h1 class="text-2xl font-bold mb-6">System Health Dashboard</h1>
  <HealthDashboard />
</div>

<ErrorModal />
