emoji scripts
=============

Tools for making or working with emojis, where emoji is used in the Slack
sense: short-code named images, up to 128x128px, intended for communication.
Not, directly, Unicode emoji code-points.

Creation of emojis, editing, uploading, syncing, whatever.

### Contributions

Contributions, both human and AI-written, welcome, as long as a human takes
responsibility for any given commit: an AI might write it, but if you submit
it, you're responsible for it.

### Licensing

ISC.


## Coding Practices

I have a strong preference for Python or Go for software which needs to be
maintained, but we'll also take robust shell if it's invoking tools such as
ImageMagick and is sufficiently short (and passes shellcheck(1)).

For Python: please include PEP 723 metadata inline in the script, so that it
can be invoked without requiring that people install everything in this
collection.  But do also ensure that you have a `main()` function which can be
used as an entrypoint for when people _do_ install all these scripts.

(He says, when there's so far 1 script).

There's an `.editorconfig` file to try to get things somewhat consistent.

I tend to use `ruff` and `ty` as LSPs when editing.  So while I might add
AI-generated code which is not 100% diagnostic-free, if I go editing then I'll
fix.  As long as you are using sensible types to have code which can be
reasoned about by those not familiar with it, this is fine for submissions.

I use `uv` and am happy to be fairly aggressive in upgrading minimum Python
versions, rather than get trapped having to support ancient Python.

If we need a task runner, it will likely be Task (<https://taskfile.dev/>).
