#!/usr/bin/env bash
set -euo pipefail

# ========= USER CONFIG (override via env on the command line) =========
: "${APP_BASE_NAME:=lucy}"            # Base name for App Runner service & ECR repo
: "${APP_ENV:=stage}"                 # "stage" | "prod"
APP_NAME="${APP_BASE_NAME}-${APP_ENV}" # Derived name: e.g. lucy-stage / lucy-prod
: "${AWS_REGION:=us-east-1}"         # e.g. us-east-1
: "${CPU:=2 vCPU}"                   # "1 vCPU" | "2 vCPU"
: "${MEMORY:=4 GB}"                  # "2 GB" | "4 GB"
: "${CONTAINER_PORT:=8000}"          # must match uvicorn port in your Dockerfile
: "${HEALTH_PATH:=/health}"          # "/" if you didn't add /health

# Extra environment variables for the container (KEY=VALUE). PORT is added automatically.
ENV_VARS=("PORT=${CONTAINER_PORT}" "AWS_REGION=${AWS_REGION}" "APP_ENV=${APP_ENV}")
# If OPENAI_API_KEY is exported in your shell, we'll add it automatically below.

# Legacy deprecation controls
: "${DEPRECATE_LEGACY:=false}"       # If true, delete legacy App Runner service
: "${LEGACY_SERVICE_NAME:=lucy}"     # Legacy single-environment service name to delete when deprecating

# VPC connector (optional) — set USE_VPC_CONNECTOR=true and pass private subnets + SG
: "${USE_VPC_CONNECTOR:=false}"
: "${SUBNET_IDS:=}"                  # "subnet-aaa subnet-bbb"
: "${SG_IDS:=}"                      # "sg-xyz"

# ========= UTILITIES =========
log()  { printf "\n\033[1;36m▶ %s\033[0m\n" "$*"; }
warn() { printf "\033[1;33m! %s\033[0m\n" "$*"; }
die()  { printf "\033[1;31m✖ %s\033[0m\n" "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "Missing dependency: $1"; }

need aws
need docker
need jq
need date

# ========= SETUP =========
export AWS_PAGER="" # don’t page CLI output

if [[ "${APP_ENV}" != "stage" && "${APP_ENV}" != "prod" ]]; then
  die "APP_ENV must be 'stage' or 'prod' (got: ${APP_ENV})"
fi

ACCOUNT_ID="$(aws sts get-caller-identity --query Account --output text 2>/dev/null || true)"
[[ -n "$ACCOUNT_ID" && "$ACCOUNT_ID" != "None" ]] || die "AWS credentials not configured (aws sts get-caller-identity failed)."

ECR_URI="${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${APP_NAME}"
IMAGE_TAG="$(date +%Y%m%d-%H%M%S)"
IMAGE_ID="${ECR_URI}:${IMAGE_TAG}"

# Add API keys from shell if present
if [[ -n "${OPENAI_API_KEY:-}" ]]; then
  ENV_VARS+=("OPENAI_API_KEY=${OPENAI_API_KEY}")
fi
if [[ -n "${ANTHROPIC_API_KEY:-}" ]]; then
  ENV_VARS+=("ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}")
fi

# Build env var MAP for App Runner (not a list!)
env_vars_map() {
  local obj="{}"
  for kv in "${ENV_VARS[@]}"; do
    local key="${kv%%=*}"
    local val="${kv#*=}"
    obj=$(jq -c --arg k "$key" --arg v "$val" '. + {($k): $v}' <<<"$obj")
  done
  printf '%s' "$obj"
}

ensure_ecr() {
  log "Ensuring ECR repo: ${APP_NAME}"
  if ! aws ecr describe-repositories \
        --region "$AWS_REGION" \
        --repository-names "$APP_NAME" >/dev/null 2>&1; then
    aws ecr create-repository \
      --region "$AWS_REGION" \
      --repository-name "$APP_NAME" >/dev/null
    log "Created ECR repo ${APP_NAME}"
  else
    log "ECR repo exists"
  fi
}

build_and_push() {
  log "Logging into ECR"
  aws ecr get-login-password --region "$AWS_REGION" \
    | docker login --username AWS --password-stdin "${ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

  log "Ensuring buildx builder"
  # create a named builder if it doesn't exist, then use it
  docker buildx inspect "${APP_NAME}-builder" >/dev/null 2>&1 \
    || docker buildx create --name "${APP_NAME}-builder" --use
  docker buildx use "${APP_NAME}-builder"

  log "Building multi-arch (amd64) image and pushing to ECR"
  docker buildx build \
    --platform linux/amd64 \
    -t "${ECR_URI}:${IMAGE_TAG}" \
    --push \
    .

  # For consistency with rest of the script
  IMAGE_ID="${ECR_URI}:${IMAGE_TAG}"
}

ensure_role() {
  local role_name="AppRunnerECRAccess-${APP_NAME}"
  local policy_arn="arn:aws:iam::aws:policy/service-role/AWSAppRunnerServicePolicyForECRAccess"

  log "Ensuring IAM role for App Runner: ${role_name}"
  if ! aws iam get-role --role-name "$role_name" >/dev/null 2>&1; then
    aws iam create-role --role-name "$role_name" --assume-role-policy-document '{
      "Version": "2012-10-17",
      "Statement": [{
        "Effect": "Allow",
        "Principal": {"Service": "build.apprunner.amazonaws.com"},
        "Action": "sts:AssumeRole"
      }]
    }' >/dev/null
    aws iam attach-role-policy --role-name "$role_name" --policy-arn "$policy_arn" >/dev/null
    log "Created role ${role_name}"
  else
    log "IAM role exists"
  fi

  ROLE_ARN="$(aws iam get-role --role-name "$role_name" --query Role.Arn --output text)"
}

ensure_vpc_connector() {
  if [[ "${USE_VPC_CONNECTOR}" != "true" ]]; then
    VPC_CONNECTOR_ARN=""
    return
  fi
  if [[ -z "${SUBNET_IDS}" || -z "${SG_IDS}" ]]; then
    warn "USE_VPC_CONNECTOR=true but SUBNET_IDS or SG_IDS not set; skipping VPC connector."
    VPC_CONNECTOR_ARN=""
    return
  fi

  local name="${APP_NAME}-vpc"
  log "Ensuring App Runner VPC connector: ${name}"

  VPC_CONNECTOR_ARN="$(aws apprunner list-vpc-connectors --region "$AWS_REGION" \
    --query "VpcConnectors[?VpcConnectorName=='${name}'].VpcConnectorArn | [0]" \
    --output text 2>/dev/null || true)"

  if [[ -z "${VPC_CONNECTOR_ARN}" || "${VPC_CONNECTOR_ARN}" == "None" ]]; then
    # Create connector (CLI accepts space-separated lists)
    VPC_CONNECTOR_ARN="$(aws apprunner create-vpc-connector \
      --region "$AWS_REGION" \
      --vpc-connector-name "$name" \
      --subnets ${SUBNET_IDS} \
      --security-groups ${SG_IDS} \
      --query VpcConnector.VpcConnectorArn --output text)"
    log "Created VPC connector: ${VPC_CONNECTOR_ARN}"
  else
    log "VPC connector exists: ${VPC_CONNECTOR_ARN}"
  fi
}

create_or_update_service() {
  local service_arn
  service_arn="$(aws apprunner list-services --region "$AWS_REGION" \
    --query "ServiceSummaryList[?ServiceName=='${APP_NAME}'].ServiceArn | [0]" \
    --output text 2>/dev/null || true)"

  local env_json; env_json="$(env_vars_map)"

  if [[ -z "${service_arn}" || "${service_arn}" == "None" ]]; then
    log "Creating App Runner service: ${APP_NAME}"
    local payload
    payload="$(jq -n \
      --arg svc "$APP_NAME" \
      --arg img "${IMAGE_ID}" \
      --arg role "${ROLE_ARN}" \
      --arg cpu "${CPU}" \
      --arg mem "${MEMORY}" \
      --arg port "${CONTAINER_PORT}" \
      --arg hp "${HEALTH_PATH}" \
      --argjson env "${env_json}" \
      --arg vpc_arn "${VPC_CONNECTOR_ARN:-}" '
      {
        ServiceName: $svc,
        SourceConfiguration: {
          ImageRepository: {
            ImageRepositoryType: "ECR",
            ImageIdentifier: $img,
            ImageConfiguration: {
              Port: $port,
              RuntimeEnvironmentVariables: $env
            }
          },
          AuthenticationConfiguration: { AccessRoleArn: $role }
        },
        InstanceConfiguration: { Cpu: $cpu, Memory: $mem },
        HealthCheckConfiguration: {
          Protocol: "HTTP",
          Path: $hp,
          Interval: 10,
          Timeout: 5,
          HealthyThreshold: 1,
          UnhealthyThreshold: 5
        }
      } | if $vpc_arn != "" then
            . + { NetworkConfiguration:
                  { EgressConfiguration: { EgressType: "VPC", VpcConnectorArn: $vpc_arn } } }
          else . end
      ')"

    local url
    url="$(aws apprunner create-service --region "$AWS_REGION" \
      --cli-input-json "$payload" \
      --query 'Service.ServiceUrl' --output text)"
    log "Service creating. URL: https://${url}"
  else
    log "Updating App Runner service image: ${APP_NAME}"
    # Update image (and keep env/port the same), OR include them explicitly:
    local update_json
    update_json="$(jq -n \
      --arg img "${IMAGE_ID}" \
      --arg port "${CONTAINER_PORT}" \
      --argjson env "${env_json}" \
      '{
        SourceConfiguration: {
          ImageRepository: {
            ImageRepositoryType: "ECR",
            ImageIdentifier: $img,
            ImageConfiguration: {
              Port: $port,
              RuntimeEnvironmentVariables: $env
            }
          }
        }
      }')"

    aws apprunner update-service --region "$AWS_REGION" \
      --service-arn "$service_arn" \
      --cli-input-json "$update_json" >/dev/null

    local url
    url="$(aws apprunner describe-service --region "$AWS_REGION" \
      --service-arn "$service_arn" \
      --query 'Service.ServiceUrl' --output text)"
    log "Service updated. URL: https://${url}"
  fi
}

deprecate_legacy_service() {
  if [[ "${DEPRECATE_LEGACY}" != "true" ]]; then
    return
  fi
  local legacy="${LEGACY_SERVICE_NAME}"
  log "Deprecating legacy App Runner service: ${legacy}"
  local legacy_arn
  legacy_arn="$(aws apprunner list-services --region "$AWS_REGION" \
    --query "ServiceSummaryList[?ServiceName=='${legacy}'].ServiceArn | [0]" \
    --output text 2>/dev/null || true)"
  if [[ -n "${legacy_arn}" && "${legacy_arn}" != "None" ]]; then
    aws apprunner delete-service --region "$AWS_REGION" \
      --service-arn "$legacy_arn" >/dev/null || true
    log "Requested deletion of legacy service: ${legacy}"
  else
    log "Legacy service '${legacy}' not found; skipping deprecation"
  fi
}

# ========= RUN =========
log "Deploying ${APP_NAME} (env=${APP_ENV}) to AWS App Runner in ${AWS_REGION}"

ensure_ecr
build_and_push
ensure_role
ensure_vpc_connector
create_or_update_service

deprecate_legacy_service

log "Done."
