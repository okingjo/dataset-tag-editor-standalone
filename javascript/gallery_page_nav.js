// Arrow-key navigation across gallery pages.
// Gradio's built-in Gallery preview cycles ArrowLeft/ArrowRight within the
// images of the current page only. This hooks the boundary cases: at the
// last image of a page ArrowRight turns to the next page (wrapping to page 1
// after the last page), at the first image ArrowLeft turns to the previous
// page (wrapping to the last page before page 1), then re-opens the preview
// on the first/last image of the new page — so arrow keys cycle through the
// ENTIRE filtered dataset with automatic page turns.
(function () {
    function setup() {
        const gallery = document.getElementById("dataset_gallery");
        if (!gallery || gallery.__dtePageNav) return !!gallery;
        gallery.__dtePageNav = true;

        let turning = false;

        function stripButtons() {
            return Array.from(gallery.querySelectorAll(".thumbnails button.thumbnail-item"));
        }
        function gridButtons() {
            return Array.from(gallery.querySelectorAll(".grid-container button.thumbnail"));
        }
        function firstStripSrc() {
            const img = gallery.querySelector(".thumbnails button.thumbnail-item img");
            return img ? img.src : "";
        }
        function pageInfo() {
            const el = document.getElementById("gallery_page_info");
            const m = el && el.textContent.match(/Page\s+(\d+)\s*\/\s*(\d+)/);
            return m ? { page: +m[1], total: +m[2] } : null;
        }
        // Current preview index: gradio focuses the matching strip thumbnail
        // on every navigation; fall back to the selected grid thumbnail.
        function currentIndex() {
            const strip = stripButtons();
            const focused = document.activeElement && document.activeElement.closest
                ? document.activeElement.closest("button.thumbnail-item") : null;
            if (focused && strip.indexOf(focused) >= 0) return strip.indexOf(focused);
            const grid = gridButtons();
            const sel = grid.findIndex(b => b.classList.contains("selected"));
            return sel;
        }
        function gotoPage(n, pick) {
            const num = document.querySelector("#gallery_goto_page_num input");
            const btn = document.getElementById("gallery_goto_page_btn");
            if (!num || !btn) { turning = false; return; }
            const before = firstStripSrc();
            num.value = String(n);
            num.dispatchEvent(new Event("input", { bubbles: true }));
            num.dispatchEvent(new Event("change", { bubbles: true }));
            btn.click();
            let tries = 0;
            const timer = setInterval(() => {
                tries++;
                const src = firstStripSrc();
                const grid = gridButtons();
                if (grid.length && src && src !== before) {
                    clearInterval(timer);
                    turning = false;
                    const target = pick === "first" ? grid[0] : grid[grid.length - 1];
                    if (target) { target.click(); target.focus(); }
                } else if (tries > 100) {
                    clearInterval(timer);
                    turning = false;
                }
            }, 100);
        }

        // Capture phase: runs before gradio's own keydown handler on the
        // .preview element, so boundary presses never cycle within the page.
        gallery.addEventListener("keydown", (e) => {
            if (e.code !== "ArrowRight" && e.code !== "ArrowLeft") return;
            if (e.altKey || e.ctrlKey || e.metaKey || e.shiftKey) return;
            if (turning) { e.preventDefault(); e.stopPropagation(); return; }
            if (!gallery.querySelector(".preview")) return; // preview closed
            const info = pageInfo();
            if (!info || info.total <= 1) return; // single page: native cycling
            const strip = stripButtons();
            if (!strip.length) return;
            const idx = currentIndex();
            if (idx < 0) return;
            const forward = e.code === "ArrowRight";
            const atEdge = forward ? idx >= strip.length - 1 : idx <= 0;
            if (!atEdge) return; // inside the page: let gradio handle it
            e.preventDefault();
            e.stopPropagation();
            turning = true;
            const target = forward
                ? (info.page >= info.total ? 1 : info.page + 1)
                : (info.page <= 1 ? info.total : info.page - 1);
            gotoPage(target, forward ? "first" : "last");
        }, true);

        return true;
    }

    // Scripts are injected into <head> before the gradio app renders; poll
    // until the gallery exists (tab switches keep the DOM, so once is enough).
    const iv = setInterval(() => { if (setup()) clearInterval(iv); }, 500);
})();
