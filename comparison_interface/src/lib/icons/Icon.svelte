<!--
  For licensing see accompanying LICENSE file.
  Copyright (C) 2024 Apple Inc. All Rights Reserved.
-->

<script lang="ts">
  export let width = "1rem";
  export let height = "1rem";
  export let focusable = false;
  export let src: string;
  export let alt: string;
</script>

<div class="svg-icon {$$props.class}">
  <img {src} {alt} />
</div>

<style>
  .svg-icon {
    display: inline-flex;
    align-self: center;
    vertical-align: middle;
  }

  .svg-icon img {
    width: 100%;
    height: 100%;
    object-fit: scale-down;
  }
</style>
