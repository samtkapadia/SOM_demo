#!/usr/bin/env bash
# Create a confirmed demo user. Usage:
#   ./scripts/create_demo_user.sh <user-pool-id> <email> <password> [region]
set -euo pipefail
POOL="${1:?user pool id}"
EMAIL="${2:?email}"
PASS="${3:?password}"
REGION="${4:-ap-southeast-2}"

aws cognito-idp admin-create-user \
  --user-pool-id "$POOL" \
  --username "$EMAIL" \
  --user-attributes Name=email,Value="$EMAIL" Name=email_verified,Value=true \
  --message-action SUPPRESS \
  --region "$REGION"

aws cognito-idp admin-set-user-password \
  --user-pool-id "$POOL" \
  --username "$EMAIL" \
  --password "$PASS" \
  --permanent \
  --region "$REGION"

echo "created $EMAIL in $POOL"
