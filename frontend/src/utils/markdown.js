/**
 * Shared markdown rendering utilities.
 *
 * Handles numbered lists, bullet lists, headings, bold text,
 * and pre-processes inline numbered lists into separate lines.
 */

/**
 * Pre-process raw text from the AI so numbered or bulleted items
 * that were crammed onto a single line get split onto their own lines.
 *
 * Patterns handled:
 *   "1. Foo 2. Bar 3. Baz"  →  "1. Foo\n2. Bar\n3. Baz"
 *   "- Foo - Bar - Baz"      →  "- Foo\n- Bar\n- Baz"
 *   "* Foo * Bar"             →  "* Foo\n* Bar"
 */
export function normalizeText(raw) {
  if (!raw) return "";

  let text = raw;

  // Split inline numbered lists.
  // Match pattern: end-of-sentence punctuation + space + digit(s) + ". "
  // e.g., "...in the soil. 2. Crop Rotation:" → "...in the soil.\n2. Crop Rotation:"
  text = text.replace(/([.!?])\s+(\d+)\.\s/g, "$1\n$2. ");

  // Also catch "word <space> <digit>. " pattern (for lists without punctuation before them)
  text = text.replace(/([a-z])\s+(\d+)\.\s/g, "$1\n$2. ");

  // Split inline unordered bullets: " - Foo" mid-line before a capital letter
  text = text.replace(/\s+-\s+(?=[A-Z])/g, "\n- ");

  // Split inline asterisk bullets
  text = text.replace(/\s+\*\s+(?=[A-Z])/g, "\n* ");

  return text;
}

/**
 * Render a markdown-ish string into React elements.
 *
 * @param {string} text - The markdown text to render.
 * @param {object} opts
 * @param {function} opts.createElement - React.createElement
 * @param {string}   [opts.classPrefix=""] - Prefix for CSS class names (e.g., "k-" for knowledge page)
 * @returns {Array} Array of React elements.
 */
export function renderMarkdown(text, opts = {}) {
  const { createElement, classPrefix = "" } = opts;
  if (!text || !createElement) return null;

  const normalized = normalizeText(text);
  const lines = normalized.split("\n");

  const cls = (name) => `${classPrefix}${name}`;

  return lines.map((line, i) => {
    const trimmed = line.trim();
    if (!trimmed) return createElement("br", { key: i });

    // Headings
    if (trimmed.startsWith("#### "))
      return createElement("h5", { key: i, className: cls("md-h") }, trimmed.slice(5));
    if (trimmed.startsWith("### "))
      return createElement("h4", { key: i, className: cls("md-h") }, trimmed.slice(4));
    if (trimmed.startsWith("## "))
      return createElement("h3", { key: i, className: cls("md-h") }, trimmed.slice(3));

    // Numbered list: "1. ", "2. ", "10. " etc.
    const numMatch = trimmed.match(/^(\d+)\.\s+(.+)/);
    if (numMatch) {
      let content = numMatch[2];
      // Process bold
      content = content.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
      // Bold the label if pattern is "Label: description"
      if (content.match(/^[^<:]+:\s/)) {
        content = content.replace(/^([^:]+):\s*/, "<strong>$1:</strong> ");
      }
      return createElement("li", {
        key: i,
        className: `${cls("md-li")} ${cls("md-li-num")}`,
        dangerouslySetInnerHTML: { __html: content },
      });
    }

    // Bullet list
    if (trimmed.startsWith("- ") || trimmed.startsWith("• ") || trimmed.startsWith("* ")) {
      const bullet = trimmed.slice(2);
      let html = bullet.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
      // Bold the label if pattern is "Label: description"
      if (html.match(/^[^<:]+:\s/)) {
        html = html.replace(/^([^:]+):\s*/, "<strong>$1:</strong> ");
      }
      return createElement("li", {
        key: i,
        className: cls("md-li"),
        dangerouslySetInnerHTML: { __html: html },
      });
    }

    // Regular paragraph with bold support
    const html = trimmed.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
    return createElement("p", {
      key: i,
      className: cls("md-p"),
      dangerouslySetInnerHTML: { __html: html },
    });
  });
}
