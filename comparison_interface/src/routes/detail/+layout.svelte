<!--
  For licensing see accompanying LICENSE file.
  Copyright (C) 2024 Apple Inc. All Rights Reserved.
-->

<script lang="ts">
  import "../../app.css";

  import { page } from "$app/stores";
  import { browser } from "$app/environment";
  import { base } from "$app/paths";

  let url: URL | null;
  $: url = browser ? $page.url : null;

  let pageName: string;
  $: if (!!url)
    pageName = url.pathname.slice(url.pathname.lastIndexOf("/") + 1);
  else pageName = "";
</script>

<main class="w-full h-screen">
  <div
    class="bg-black opacity-90 px-3 py-3 h-12 w-full fixed flex justify-between text-white"
  >
    <div>
      <a
        class="mr-6 hover:opacity-70 font-bold"
        href="{base}/{!!url && url.searchParams.has('models')
          ? '?selection=' + url.searchParams.get('models')
          : ''}">&lsaquo; Back to Overview</a
      >
      <a
        class="mr-3 transition-opacity duration-200 {pageName == 'instances'
          ? 'opacity-70'
          : 'hover:opacity-70'}"
        href="{base}/detail/instances{url ? url.search : ''}">Behaviors</a
      >
      <a
        class="mr-3 transition-opacity duration-200 {pageName == 'tensors'
          ? 'opacity-70'
          : 'hover:opacity-70'}"
        href="{base}/detail/tensors{url ? url.search : ''}">Layers</a
      >
    </div>
    <a class="block mr-3 hover:opacity-70" href="{base}/settings"
      >Connect to Model...</a
    >
  </div>

  <slot />
</main>
