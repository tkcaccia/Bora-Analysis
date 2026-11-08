document.querySelectorAll('.geojson-download').forEach(button => {
  button.addEventListener('click', async () => {
    const label = button.textContent;
    const status = document.getElementById(button.getAttribute('aria-describedby'));
    button.disabled = true;
    button.setAttribute('aria-busy', 'true');
    status.textContent = 'Preparing download…';
    try {
      const response = await fetch(button.dataset.src);
      if (!response.ok) throw new Error('Download unavailable');
      const data = await response.arrayBuffer();
      const bytes = new Uint8Array(data);
      let blob = new Blob([data], {type: 'application/geo+json'});
      if (bytes[0] === 0x1f && bytes[1] === 0x8b) {
        if (!('DecompressionStream' in window)) throw new Error('Use compressed copy');
        blob = await new Response(blob.stream().pipeThrough(new DecompressionStream('gzip'))).blob();
      }
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = button.dataset.filename;
      document.body.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 60000);
      status.textContent = 'Download ready.';
    } catch (error) {
      status.textContent = 'Use the compressed copy beside this button.';
    } finally {
      button.disabled = false;
      button.removeAttribute('aria-busy');
      button.textContent = label;
    }
  });
});
