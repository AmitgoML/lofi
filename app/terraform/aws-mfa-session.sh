#!/usr/bin/env bash
# aws-mfa-session.sh
# Create/refresh an AWS CLI session profile using MFA.
# Requires: AWS CLI v2, an existing base profile with long-lived keys or SSO.
# Usage:
#   ./aws-mfa-session.sh -p myuser [-t myuser-session] [-s arn:aws:iam::123456789012:mfa/alex] [-d 43200] [-r us-east-1] [MFA_CODE]
#
# Then use:
#   aws --profile myuser-session s3 ls
#   TF_VAR: provider "aws" { profile = "myuser-session" }

set -euo pipefail

BASE_PROFILE=""
TARGET_PROFILE=""
MFA_SERIAL=""
DURATION="43200"  # 12 hours (max is usually 36,000 for GetSessionToken; some orgs allow 43,200)
REGION=""
PRINT_EXPORTS="false"

usage() {
  cat <<EOF
Usage: $(basename "$0") -p BASE_PROFILE [options] [MFA_CODE]

Options:
  -p  Base profile in ~/.aws/{config,credentials} (required)
  -t  Target session profile name (default: lofi-mfa-session)
  -s  MFA device ARN (e.g., arn:aws:iam::123456789012:mfa/your-user)
      If omitted, will try to read from ~/.aws/config (profile.<BASE_PROFILE or <BASE_PROFILE>-mfa>.mfa_serial)
  -d  Session duration seconds (default: 43200)
  -r  Region to write to target profile (defaults to base profile's region)
  -e  Print 'export AWS_*' env vars instead of writing a profile (no file changes)

Examples:
  $(basename "$0") -p myuser 123456
  $(basename "$0") -p myuser -t tf-session -s arn:aws:iam::123:mfa/alex -d 36000
EOF
}

while getopts ":p:t:s:d:r:eh" opt; do
  case $opt in
    p) BASE_PROFILE="$OPTARG" ;;
    t) TARGET_PROFILE="$OPTARG" ;;
    s) MFA_SERIAL="$OPTARG" ;;
    d) DURATION="$OPTARG" ;;
    r) REGION="$OPTARG" ;;
    e) PRINT_EXPORTS="true" ;;
    h) usage; exit 0 ;;
    \?) echo "Invalid option: -$OPTARG" >&2; usage; exit 2 ;;
    :)  echo "Option -$OPTARG requires an argument." >&2; usage; exit 2 ;;
  esac
done
shift $((OPTIND-1))

MFA_CODE="${1:-}"

if [[ -z "$BASE_PROFILE" ]]; then
  echo "Error: -p BASE_PROFILE is required." >&2
  usage
  exit 2
fi

if [[ -z "${TARGET_PROFILE}" ]]; then
  TARGET_PROFILE="lofi-mfa-session"
fi

# Basic checks
command -v aws >/dev/null 2>&1 || { echo "aws CLI not found. Install AWS CLI v2."; exit 1; }

# Validate base profile can call STS (also sets default region if we can read it)
echo "🔎 Checking base profile '$BASE_PROFILE'..."
aws sts get-caller-identity --profile "$BASE_PROFILE" >/dev/null

# Infer REGION if not provided
if [[ -z "$REGION" ]]; then
  REGION="$(aws configure get region --profile "$BASE_PROFILE" || true)"
fi
if [[ -z "$REGION" ]]; then
  REGION="us-east-1"
fi

# Infer MFA serial if not provided: try profile.<BASE>.mfa_serial then profile.<BASE>-mfa.mfa_serial
if [[ -z "$MFA_SERIAL" ]]; then
  MFA_SERIAL="$(aws configure get "profile.${BASE_PROFILE}.mfa_serial" || true)"
  if [[ -z "$MFA_SERIAL" ]]; then
    MFA_SERIAL="$(aws configure get "profile.${BASE_PROFILE}-mfa.mfa_serial" || true)"
  fi
fi

if [[ -z "$MFA_SERIAL" ]]; then
  echo "Error: Could not determine MFA serial ARN. Use -s or set mfa_serial in ~/.aws/config." >&2
  echo "Example config:"
  echo "[profile ${BASE_PROFILE}]" >&2
  echo "mfa_serial = arn:aws:iam::034362062829:mfa/Alex" >&2
  exit 2
fi

# Prompt for MFA code if not provided
if [[ -z "$MFA_CODE" ]]; then
  read -r -p "Enter 6-digit MFA code for ${MFA_SERIAL}: " MFA_CODE
fi

echo "🔐 Requesting session token (duration=${DURATION}s, region=${REGION})..."
# Get session credentials as tab-separated text: AK SK ST EXP
STS_OUT="$(aws sts get-session-token \
  --profile "$BASE_PROFILE" \
  --serial-number "$MFA_SERIAL" \
  --token-code "$MFA_CODE" \
  --duration-seconds "$DURATION" \
  --query 'Credentials.[AccessKeyId,SecretAccessKey,SessionToken,Expiration]' \
  --output text)"

if [[ -z "$STS_OUT" ]]; then
  echo "Error: STS did not return credentials. Check your code and MFA device." >&2
  exit 1
fi

# shellcheck disable=SC2086
read -r AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_SESSION_TOKEN EXPIRATION <<< "$STS_OUT"

if [[ "$PRINT_EXPORTS" == "true" ]]; then
  cat <<EOF
export AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
export AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
export AWS_SESSION_TOKEN=${AWS_SESSION_TOKEN}
export AWS_DEFAULT_REGION=${REGION}
# Expires at: ${EXPIRATION}
EOF
  echo "✅ Session credentials printed as export statements. (Expires: ${EXPIRATION})"
  exit 0
fi

echo "📝 Writing session credentials to profile '${TARGET_PROFILE}'..."
aws configure set aws_access_key_id     "$AWS_ACCESS_KEY_ID"     --profile "$TARGET_PROFILE"
aws configure set aws_secret_access_key "$AWS_SECRET_ACCESS_KEY" --profile "$TARGET_PROFILE"
aws configure set aws_session_token     "$AWS_SESSION_TOKEN"     --profile "$TARGET_PROFILE"
aws configure set region                "$REGION"                --profile "$TARGET_PROFILE"

# Quick sanity check
echo "🔁 Verifying with 'aws sts get-caller-identity' using '${TARGET_PROFILE}'..."
aws sts get-caller-identity --profile "$TARGET_PROFILE" >/dev/null

echo "✅ Done. Session written to profile '${TARGET_PROFILE}'."
echo "   Expires at: ${EXPIRATION}"
echo "   Use it like: aws --profile ${TARGET_PROFILE} s3 ls"

