/* Visual key-mapping canvas for the Merge page.
 *
 * Draws curved SVG paths between left/right column-pills to reflect the
 * current list of key pairs (stored in `merge-keys-store`). Colour-codes
 * by dtype compatibility: solid accent-green = types match, dashed amber
 * = type mismatch (requires casting).
 *
 * This module exposes `window.kbDrawMergeKeys(pairs, dtypes)`, which is
 * invoked by a Dash clientside_callback when the store changes. A resize
 * listener re-runs the last draw to keep lines aligned on window resize.
 */
(function () {
  var last = { pairs: [], dtypes: { left: {}, right: {} } };

  function buildPath(x1, y1, x2, y2) {
    // Horizontal cubic Bezier — two control points at the midpoint X, each
    // carrying its endpoint's Y. Produces a gentle S-curve like the mockup.
    var cx = (x1 + x2) / 2;
    return "M" + x1 + "," + y1 + " C" + cx + "," + y1 + " " + cx + "," + y2 + " " + x2 + "," + y2;
  }

  // Build a {side+col -> element} index over all pill DOM nodes on the page.
  // Dash serialises pattern-match dict IDs as JSON (alphabetical key order),
  // e.g. `id='{"col":"client_id","side":"left","type":"mkp"}'`. We find them
  // with a substring match then JSON.parse the id to dispatch exactly.
  function buildPillIndex() {
    var idx = { left: {}, right: {} };
    var candidates = document.querySelectorAll('[id*="mkp"]');
    for (var i = 0; i < candidates.length; i++) {
      var raw = candidates[i].id;
      if (!raw || raw.charAt(0) !== "{") continue;
      try {
        var p = JSON.parse(raw);
        if (p && p.type === "mkp" && (p.side === "left" || p.side === "right")) {
          idx[p.side][p.col] = candidates[i];
        }
      } catch (e) { /* not one of our ids */ }
    }
    return idx;
  }

  function draw() {
    var host = document.getElementById("merge-keys-svg-host");
    if (!host) return;
    var cRect = host.getBoundingClientRect();
    var w = cRect.width;
    var h = cRect.height;
    if (w <= 0 || h <= 0) return;

    var pairs = last.pairs || [];
    var dtypes = last.dtypes || { left: {}, right: {} };
    var ltypes = dtypes.left || {};
    var rtypes = dtypes.right || {};

    var pillIdx = buildPillIndex();
    var parts = [];
    for (var i = 0; i < pairs.length; i++) {
      var p = pairs[i];
      var Lel = pillIdx.left[p.left];
      var Rel = pillIdx.right[p.right];
      if (!Lel || !Rel) continue;
      var l = Lel.getBoundingClientRect();
      var r = Rel.getBoundingClientRect();
      var x1 = l.right - cRect.left;
      var y1 = l.top + l.height / 2 - cRect.top;
      var x2 = r.left - cRect.left;
      var y2 = r.top + r.height / 2 - cRect.top;
      var okType = ltypes[p.left] === rtypes[p.right];
      var cls = okType
        ? "kb-keys-link kb-keys-link--ok"
        : "kb-keys-link kb-keys-link--mismatch";
      var dotCls = okType ? "kb-keys-link--ok-dot" : "kb-keys-link--mismatch-dot";
      parts.push(
        '<path class="' + cls + '" d="' + buildPath(x1, y1, x2, y2) + '"/>' +
        '<circle class="' + dotCls + '" cx="' + x2 + '" cy="' + y2 + '" r="3"/>'
      );
    }

    host.innerHTML =
      '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 ' + w + " " + h +
      '" width="' + w + '" height="' + h + '">' + parts.join("") + "</svg>";
  }

  window.kbDrawMergeKeys = function (pairs, dtypes) {
    if (pairs !== undefined && pairs !== null) last.pairs = pairs;
    if (dtypes !== undefined && dtypes !== null) last.dtypes = dtypes;
    // Wait one frame so the pill DOM from the last Python render is committed.
    requestAnimationFrame(function () {
      requestAnimationFrame(draw);
    });
  };

  // Keep the canvas aligned when the window resizes.
  window.addEventListener("resize", function () {
    requestAnimationFrame(draw);
  });

  // First paint after the page loads (in case the store already has pairs).
  document.addEventListener("DOMContentLoaded", function () {
    requestAnimationFrame(draw);
  });
})();
