<!--
  For licensing see accompanying LICENSE file.
  Copyright (C) 2024 Apple Inc. All Rights Reserved.
-->

<script lang="ts">
  import "../../app.css";
  import { page } from "$app/stores";
  import { base } from "$app/paths";
</script>

<main class="w-full h-screen">
  <div
    class="bg-black opacity-90 px-3 py-3 h-12 w-full fixed flex justify-between text-white"
  >
    <div class="font-bold">
      <a class="mr-3 hover:opacity-70" href="{base}/">Compress and Compare</a>
    </div>
    {#if !$page.url.pathname.endsWith("settings")}
      <div class="text-sm">
        <a class="mr-3 hover:opacity-70" href="{base}/settings"
          >Connect to Model...</a
        >
      </div>
    {/if}
  </div>

  <slot />
</main>
