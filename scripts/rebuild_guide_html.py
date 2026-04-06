"""
Rebuild all custom HTML components in guide markdown files.
Strips corrupted HTML and rebuilds with correct structure.

Understanding of the corruption:
  The original structure had these patterns:

  Pattern A (with code-body wrapper):
    <span class="filename">FNAME</span>    <- orphaned
    </div>                                  <- stray close
    <div class="code-body">                 <- extra open
    <div class="code-window">               <- open
    <div class="code-header">               <- open (no close, but 2 </div> at end handle it)
    <div class="dots">...</div>             <- balanced
    ```code```
    </div>                                  <- closes code-header
    </div>                                  <- closes code-window

  Pattern B (without code-body):
    <span class="filename">FNAME</span>    <- orphaned
    </div>                                  <- stray close
    <div class="code-window">               <- open
    <div class="code-header">               <- open
    <div class="dots">...</div>             <- balanced
    ```code```
    </div>                                  <- closes code-header
    </div>                                  <- closes code-window

  Fix strategy:
    1. Remove orphaned filename + stray </div> + <div class="code-body">
    2. Move filename inside code-header (after dots)
    3. Ensure blank line between dots div (or filename) and code fence
    4. Do NOT add extra </div> for code-header (existing ones handle it)
"""
import re
import glob
import os
import sys


def rebuild_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    original = content

    # =========================================================================
    # STEP 1: Fix Pattern A - orphaned filename + </div> + code-body + code-window
    # =========================================================================
    # Remove orphan + stray close + code-body open, move filename into header
    content = re.sub(
        r'<span class="filename">(.*?)</span>\s*\n'
        r'</div>\s*\n'
        r'<div class="code-body">\s*\n'
        r'\s*\n?'
        r'(<div class="code-window">\s*\n'
        r'<div class="code-header">\s*\n'
        r'<div class="dots">.*?</div>)\s*\n'
        r'\s*\n?'
        r'(```)',
        lambda m: (
            f'{m.group(2)}\n'
            f'<span class="filename">{m.group(1)}</span>\n'
            f'\n'
            f'{m.group(3)}'
        ),
        content
    )

    # =========================================================================
    # STEP 2: Fix Pattern B - orphaned filename + </div> + blank + code-window
    # =========================================================================
    content = re.sub(
        r'<span class="filename">(.*?)</span>\s*\n'
        r'</div>\s*\n'
        r'\s*\n'
        r'(<div class="code-window">\s*\n'
        r'<div class="code-header">\s*\n'
        r'<div class="dots">.*?</div>)\s*\n'
        r'\s*\n?'
        r'(```)',
        lambda m: (
            f'{m.group(2)}\n'
            f'<span class="filename">{m.group(1)}</span>\n'
            f'\n'
            f'{m.group(3)}'
        ),
        content
    )

    # =========================================================================
    # STEP 3: Fix remaining orphaned filename spans
    # =========================================================================
    content = re.sub(
        r'\n?<span class="filename">.*?</span>\s*\n</div>\s*\n',
        '\n',
        content
    )

    # =========================================================================
    # STEP 4: Remove any remaining <div class="code-body"> and its closing </div>
    # =========================================================================
    # First handle: <div class="code-body"> that might still exist
    # Also need to remove the corresponding </div>
    # The code-body </div> is one of the two at the end of the block
    # Since removing code-body open means we have 1 extra close, we need to remove one </div>
    # Strategy: remove the code-body open, then find blocks with 3+ consecutive </div> and reduce
    content = re.sub(
        r'<div class="code-body">\s*\n',
        '',
        content
    )

    # =========================================================================
    # STEP 5: Ensure blank line between dots/filename and code fence
    # =========================================================================
    # Fix: dots div immediately followed by code fence (no blank line)
    content = re.sub(
        r'(<div class="dots">.*?</div>)\s*\n(```)',
        r'\1\n\n\2',
        content
    )
    # Fix: filename span immediately followed by code fence (no blank line)
    content = re.sub(
        r'(<span class="filename">.*?</span>)\s*\n(```)',
        r'\1\n\n\2',
        content
    )

    # =========================================================================
    # STEP 6: Remove bare "example.py" lines
    # =========================================================================
    content = re.sub(r'^example\.py\s*$', '', content, flags=re.MULTILINE)

    # =========================================================================
    # STEP 7: Fix callout divs trapped inside code fences
    # =========================================================================
    def fix_callouts_in_fences(content):
        """Extract callout divs that ended up inside code fences.

        Close the fence before the callout, output the callout, then check
        if there's more code after and reopen the fence if needed.
        """
        lines = content.split('\n')
        result = []
        in_fence = False
        fence_marker = '```'
        fence_lang = ''
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            if stripped.startswith('```') or stripped.startswith('````'):
                backticks = ''
                for ch in stripped:
                    if ch == '`':
                        backticks += ch
                    else:
                        break

                if not in_fence:
                    in_fence = True
                    fence_marker = backticks
                    fence_lang = stripped[len(backticks):]
                    result.append(line)
                    i += 1
                    continue
                elif backticks == fence_marker and (
                    stripped == backticks or len(stripped.replace('`', '').strip()) == 0
                ):
                    in_fence = False
                    result.append(line)
                    i += 1
                    continue

            if in_fence and stripped.startswith('<div class="callout-'):
                result.append(fence_marker)  # close fence
                result.append('')
                # Collect the callout block
                callout_lines = [line]
                i += 1
                while i < len(lines):
                    cl = lines[i]
                    callout_lines.append(cl)
                    if cl.strip() == '</div>':
                        i += 1
                        break
                    i += 1
                for cl in callout_lines:
                    result.append(cl)
                result.append('')
                # Skip blank lines after callout
                while i < len(lines) and lines[i].strip() == '':
                    i += 1
                # Check if there's more code content before the closing fence
                # Look ahead to see if there's non-fence content before a closing fence
                has_more_code = False
                for j in range(i, len(lines)):
                    sj = lines[j].strip()
                    if sj == '' :
                        continue
                    if sj.startswith(fence_marker) and len(sj.replace('`', '').strip()) == 0:
                        # Found closing fence marker
                        has_more_code = False
                        break
                    else:
                        # Found non-blank, non-fence content
                        has_more_code = True
                        break
                if has_more_code:
                    # Reopen fence with same language
                    result.append(f'{fence_marker}{fence_lang}')
                    # fence is still open
                else:
                    in_fence = False
                continue

            result.append(line)
            i += 1

        return '\n'.join(result)

    content = fix_callouts_in_fences(content)

    # =========================================================================
    # STEP 8: Ensure callout divs have blank lines for markdown rendering
    # =========================================================================
    content = re.sub(
        r'(<div class="callout-\w+">)\n(?!\n)',
        r'\1\n\n',
        content
    )

    # =========================================================================
    # STEP 9: Clean up multiple blank lines (max 2 consecutive)
    # =========================================================================
    content = re.sub(r'\n{4,}', '\n\n\n', content)

    # =========================================================================
    # STEP 10: Ensure headings have preceding blank line
    # =========================================================================
    content = re.sub(r'([^\n])\n(#{1,3} )', r'\1\n\n\2', content)

    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    files = sorted(glob.glob('courses/**/guides/*.md', recursive=True))
    files = [f for f in files if '_slides' not in f]

    print(f"Found {len(files)} guide files to process")

    fixed = 0
    for filepath in files:
        if rebuild_file(filepath):
            fixed += 1

    print(f"Rebuilt HTML in {fixed}/{len(files)} files")


if __name__ == '__main__':
    main()
