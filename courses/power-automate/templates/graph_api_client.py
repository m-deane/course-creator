"""
Microsoft Graph API Client - Copy and customize for your Power Automate integrations
Works with: MSAL + Microsoft Graph API v1.0
Time to working: 10 minutes

Usage:
    # Client credentials (app-only, no user sign-in)
    client = GraphApiClient.from_client_credentials(
        tenant_id=os.environ["AZURE_TENANT_ID"],
        client_id=os.environ["AZURE_CLIENT_ID"],
        client_secret=os.environ["AZURE_CLIENT_SECRET"],
    )

    # Delegated (user context, requires interactive login or refresh token)
    client = GraphApiClient.from_delegated(
        tenant_id=os.environ["AZURE_TENANT_ID"],
        client_id=os.environ["AZURE_CLIENT_ID"],
        username=os.environ["AZURE_USERNAME"],
        password=os.environ["AZURE_PASSWORD"],
    )

    flows = client.list_flows(environment_id="Default-...")
    for flow in flows:
        print(flow["name"])
"""

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import msal
import requests

# ============================================================
# CUSTOMIZE THESE
# ============================================================

AZURE_TENANT_ID: str = os.environ.get("AZURE_TENANT_ID", "YOUR_TENANT_ID")
AZURE_CLIENT_ID: str = os.environ.get("AZURE_CLIENT_ID", "YOUR_CLIENT_ID")
AZURE_CLIENT_SECRET: str = os.environ.get("AZURE_CLIENT_SECRET", "YOUR_CLIENT_SECRET")

# Power Automate environment — find in Power Automate > Settings > Session details
DEFAULT_ENVIRONMENT_ID: str = os.environ.get(
    "POWER_AUTOMATE_ENVIRONMENT_ID", "Default-YOUR_TENANT_ID"
)

# Retry configuration
MAX_RETRIES: int = 3
RETRY_BACKOFF_BASE: float = 2.0   # seconds; wait = base ** attempt
RETRY_STATUS_CODES: tuple[int, ...] = (429, 500, 502, 503, 504)

# ============================================================
# COPY THIS ENTIRE BLOCK (production-ready)
# ============================================================

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"
FLOW_API_BASE_URL = "https://api.flow.microsoft.com/providers/Microsoft.ProcessSimple"

# Scopes required for Power Automate flow management
FLOW_SCOPES = ["https://service.flow.microsoft.com/.default"]
GRAPH_SCOPES = ["https://graph.microsoft.com/.default"]


@dataclass
class FlowRun:
    """Represents a single Power Automate flow run."""

    run_id: str
    status: str                     # "Running", "Succeeded", "Failed", "Cancelled"
    start_time: str
    end_time: Optional[str]
    trigger_name: str
    properties: dict[str, Any] = field(default_factory=dict)


@dataclass
class Flow:
    """Represents a Power Automate cloud flow."""

    flow_id: str
    name: str
    display_name: str
    state: str                      # "Started", "Stopped", "Suspended"
    created_time: str
    last_modified_time: str
    trigger_type: str
    properties: dict[str, Any] = field(default_factory=dict)


class GraphApiError(Exception):
    """Raised when a Graph API or Flow API call fails after all retries."""

    def __init__(self, message: str, status_code: int, response_body: str):
        super().__init__(message)
        self.status_code = status_code
        self.response_body = response_body


class GraphApiClient:
    """
    Production-ready client for Microsoft Graph API and Power Automate Flow API.

    Supports both client-credentials (app-only) and delegated (user context) auth flows
    via MSAL. Tokens are cached and refreshed automatically.

    Example:
        client = GraphApiClient.from_client_credentials(
            tenant_id="...", client_id="...", client_secret="..."
        )
        flows = client.list_flows(environment_id="Default-...")
        run = client.trigger_flow(environment_id="...", flow_id="...", body={})
    """

    def __init__(self, msal_app: msal.ClientApplication, scopes: list[str]):
        self._msal_app = msal_app
        self._scopes = scopes
        self._session = requests.Session()
        self._session.headers.update({"Content-Type": "application/json"})

    # ------------------------------------------------------------------
    # Factory constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_client_credentials(
        cls,
        tenant_id: str,
        client_id: str,
        client_secret: str,
    ) -> "GraphApiClient":
        """
        Create a client using the OAuth2 client-credentials flow (app-only auth).
        No user interaction required. Suitable for background automation.

        Args:
            tenant_id: Azure AD tenant ID (GUID).
            client_id: Application (client) ID registered in Azure AD.
            client_secret: Client secret for the registered application.

        Returns:
            Configured GraphApiClient instance.
        """
        authority = f"https://login.microsoftonline.com/{tenant_id}"
        app = msal.ConfidentialClientApplication(
            client_id=client_id,
            client_credential=client_secret,
            authority=authority,
        )
        logger.info("GraphApiClient initialised with client-credentials flow")
        return cls(msal_app=app, scopes=FLOW_SCOPES)

    @classmethod
    def from_delegated(
        cls,
        tenant_id: str,
        client_id: str,
        username: str,
        password: str,
    ) -> "GraphApiClient":
        """
        Create a client using the OAuth2 ROPC (Resource Owner Password Credentials) flow.
        Requires a user account. Suitable for automation where interactive login is not
        possible and MFA is disabled for the service account.

        Args:
            tenant_id: Azure AD tenant ID (GUID).
            client_id: Application (client) ID registered in Azure AD.
            username: UPN of the user account (e.g. svc-automate@contoso.com).
            password: Password for the user account.

        Returns:
            Configured GraphApiClient instance.
        """
        authority = f"https://login.microsoftonline.com/{tenant_id}"
        app = msal.PublicClientApplication(client_id=client_id, authority=authority)
        # Acquire token immediately to surface auth errors at construction time
        scopes = [
            "https://service.flow.microsoft.com/Flows.Read.All",
            "https://service.flow.microsoft.com/Flows.Manage.All",
        ]
        result = app.acquire_token_by_username_password(
            username=username, password=password, scopes=scopes
        )
        if "error" in result:
            raise GraphApiError(
                f"Delegated auth failed: {result.get('error_description')}",
                status_code=401,
                response_body=str(result),
            )
        logger.info("GraphApiClient initialised with delegated (ROPC) flow")
        return cls(msal_app=app, scopes=scopes)

    # ------------------------------------------------------------------
    # Token management
    # ------------------------------------------------------------------

    def _get_token(self) -> str:
        """
        Return a valid bearer token, using the MSAL cache when possible.
        MSAL handles expiry and silent refresh automatically.
        """
        # Try silent refresh first (uses cache)
        accounts = self._msal_app.get_accounts()
        result = None
        if accounts:
            result = self._msal_app.acquire_token_silent(self._scopes, account=accounts[0])

        # Fall back to a fresh token (client credentials)
        if not result:
            result = self._msal_app.acquire_token_for_client(scopes=self._scopes)

        if "access_token" not in result:
            raise GraphApiError(
                f"Token acquisition failed: {result.get('error_description', 'unknown error')}",
                status_code=401,
                response_body=str(result),
            )
        return result["access_token"]

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _request(
        self,
        method: str,
        url: str,
        *,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Make an authenticated HTTP request with automatic retry on transient errors.

        Retries on RETRY_STATUS_CODES using exponential backoff.
        Raises GraphApiError on persistent failure.
        """
        token = self._get_token()
        headers = {"Authorization": f"Bearer {token}"}

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = self._session.request(
                    method=method,
                    url=url,
                    headers=headers,
                    params=params,
                    json=json_body,
                    timeout=30,
                )

                if response.status_code in RETRY_STATUS_CODES and attempt < MAX_RETRIES:
                    wait = RETRY_BACKOFF_BASE ** attempt
                    logger.warning(
                        "HTTP %s from %s (attempt %d/%d); retrying in %.1fs",
                        response.status_code,
                        url,
                        attempt,
                        MAX_RETRIES,
                        wait,
                    )
                    time.sleep(wait)
                    continue

                if not response.ok:
                    raise GraphApiError(
                        f"HTTP {response.status_code} from {url}",
                        status_code=response.status_code,
                        response_body=response.text,
                    )

                # 204 No Content returns empty body
                return response.json() if response.content else {}

            except requests.RequestException as exc:
                if attempt == MAX_RETRIES:
                    raise GraphApiError(
                        f"Request failed after {MAX_RETRIES} attempts: {exc}",
                        status_code=0,
                        response_body=str(exc),
                    ) from exc
                wait = RETRY_BACKOFF_BASE ** attempt
                logger.warning("Request error (attempt %d/%d): %s; retrying in %.1fs", attempt, MAX_RETRIES, exc, wait)
                time.sleep(wait)

        # Should not be reached
        raise GraphApiError("Exhausted retries", status_code=0, response_body="")

    def _paginate(self, url: str, params: Optional[dict[str, Any]] = None) -> list[dict[str, Any]]:
        """
        Collect all pages from a Graph API list endpoint using @odata.nextLink.

        Args:
            url: Initial request URL.
            params: Optional OData query parameters ($filter, $select, $top, etc.).

        Returns:
            Flat list of all items across all pages.
        """
        items: list[dict[str, Any]] = []
        next_url: Optional[str] = url

        while next_url:
            data = self._request("GET", next_url, params=params)
            items.extend(data.get("value", []))
            next_url = data.get("@odata.nextLink")
            params = None  # nextLink already encodes query params
            logger.debug("Paginated %d items so far", len(items))

        return items

    # ------------------------------------------------------------------
    # Power Automate Flow API methods
    # ------------------------------------------------------------------

    def list_flows(
        self,
        environment_id: str = DEFAULT_ENVIRONMENT_ID,
        *,
        filter_state: Optional[str] = None,
    ) -> list[Flow]:
        """
        List all cloud flows in a Power Automate environment.

        Args:
            environment_id: The environment GUID (e.g. "Default-<tenant-id>").
                            Find it in Power Automate > Settings > Session details.
            filter_state: Optional state filter — "Started", "Stopped", or "Suspended".

        Returns:
            List of Flow dataclass instances.

        Example:
            flows = client.list_flows("Default-abc123")
            started = client.list_flows("Default-abc123", filter_state="Started")
        """
        url = f"{FLOW_API_BASE_URL}/environments/{environment_id}/flows"
        params: dict[str, Any] = {"api-version": "2016-11-01"}
        if filter_state:
            params["$filter"] = f"properties/state eq '{filter_state}'"

        raw_items = self._paginate(url, params=params)
        flows = []
        for item in raw_items:
            props = item.get("properties", {})
            flows.append(
                Flow(
                    flow_id=item["name"],
                    name=item["name"],
                    display_name=props.get("displayName", ""),
                    state=props.get("state", ""),
                    created_time=props.get("createdTime", ""),
                    last_modified_time=props.get("lastModifiedTime", ""),
                    trigger_type=props.get("definitionSummary", {}).get("triggers", [{}])[0].get("type", ""),
                    properties=props,
                )
            )
        logger.info("Listed %d flows in environment %s", len(flows), environment_id)
        return flows

    def get_flow(self, environment_id: str, flow_id: str) -> Flow:
        """
        Retrieve details for a single cloud flow.

        Args:
            environment_id: The environment GUID.
            flow_id: The flow GUID (shown in the flow URL in Power Automate).

        Returns:
            Flow dataclass with full properties.

        Example:
            flow = client.get_flow("Default-abc123", "flow-guid-here")
            print(flow.display_name, flow.state)
        """
        url = f"{FLOW_API_BASE_URL}/environments/{environment_id}/flows/{flow_id}"
        data = self._request("GET", url, params={"api-version": "2016-11-01"})
        props = data.get("properties", {})
        return Flow(
            flow_id=data["name"],
            name=data["name"],
            display_name=props.get("displayName", ""),
            state=props.get("state", ""),
            created_time=props.get("createdTime", ""),
            last_modified_time=props.get("lastModifiedTime", ""),
            trigger_type=props.get("definitionSummary", {}).get("triggers", [{}])[0].get("type", ""),
            properties=props,
        )

    def trigger_flow(
        self,
        environment_id: str,
        flow_id: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Manually trigger a cloud flow that has an HTTP Request trigger.

        The flow must be configured with a manual (HTTP Request) trigger and the
        caller must have the flow URL. This method posts to that callback URL.

        Args:
            environment_id: The environment GUID.
            flow_id: The flow GUID.
            body: JSON body passed to the flow trigger as inputs.
                  Must match the schema defined in the flow's trigger step.

        Returns:
            Raw response dict from the trigger endpoint (may be empty for async flows).

        Example:
            result = client.trigger_flow(
                environment_id="Default-abc123",
                flow_id="flow-guid-here",
                body={"item_id": 42, "action": "approve"},
            )
        """
        # The trigger endpoint is the flow's callback URL embedded in its definition.
        # For automation, retrieve it from the flow properties first.
        flow = self.get_flow(environment_id, flow_id)
        trigger_url = (
            flow.properties
            .get("definitionSummary", {})
            .get("triggers", [{}])[0]
            .get("metadata", {})
            .get("operationMetadataId", None)
        )
        if not trigger_url:
            # Fall back to the standard run endpoint
            url = (
                f"{FLOW_API_BASE_URL}/environments/{environment_id}"
                f"/flows/{flow_id}/triggers/manual/run"
            )
            result = self._request("POST", url, params={"api-version": "2016-11-01"}, json_body=body)
        else:
            result = self._request("POST", trigger_url, json_body=body)

        logger.info("Triggered flow %s in environment %s", flow_id, environment_id)
        return result

    def get_flow_runs(
        self,
        environment_id: str,
        flow_id: str,
        *,
        top: int = 50,
        filter_status: Optional[str] = None,
    ) -> list[FlowRun]:
        """
        List recent run history for a cloud flow.

        Args:
            environment_id: The environment GUID.
            flow_id: The flow GUID.
            top: Maximum number of runs to return (default 50, max 250).
            filter_status: Optional status filter — "Running", "Succeeded", "Failed", "Cancelled".

        Returns:
            List of FlowRun instances, newest first.

        Example:
            runs = client.get_flow_runs("Default-abc123", "flow-guid", filter_status="Failed")
            for run in runs:
                print(run.run_id, run.status, run.start_time)
        """
        url = (
            f"{FLOW_API_BASE_URL}/environments/{environment_id}"
            f"/flows/{flow_id}/runs"
        )
        params: dict[str, Any] = {"api-version": "2016-11-01", "$top": top}
        if filter_status:
            params["$filter"] = f"status eq '{filter_status}'"

        raw_items = self._paginate(url, params=params)
        runs = []
        for item in raw_items:
            props = item.get("properties", {})
            runs.append(
                FlowRun(
                    run_id=item["name"],
                    status=props.get("status", ""),
                    start_time=props.get("startTime", ""),
                    end_time=props.get("endTime"),
                    trigger_name=props.get("trigger", {}).get("name", ""),
                    properties=props,
                )
            )
        logger.info("Retrieved %d runs for flow %s", len(runs), flow_id)
        return runs

    def get_flow_run_details(
        self,
        environment_id: str,
        flow_id: str,
        run_id: str,
    ) -> dict[str, Any]:
        """
        Get detailed information about a specific flow run, including per-action results.

        Args:
            environment_id: The environment GUID.
            flow_id: The flow GUID.
            run_id: The run GUID (from get_flow_runs()).

        Returns:
            Raw dict with full run details including action-level status and outputs.

        Example:
            run_id = runs[0].run_id
            details = client.get_flow_run_details("Default-abc123", "flow-guid", run_id)
            for action_name, action_data in details.get("actions", {}).items():
                print(action_name, action_data["status"])
        """
        url = (
            f"{FLOW_API_BASE_URL}/environments/{environment_id}"
            f"/flows/{flow_id}/runs/{run_id}"
        )
        data = self._request("GET", url, params={"api-version": "2016-11-01"})
        logger.info("Retrieved run details for run %s (flow %s)", run_id, flow_id)
        return data

    # ------------------------------------------------------------------
    # Microsoft Graph API convenience methods
    # ------------------------------------------------------------------

    def get_user(self, user_id_or_upn: str) -> dict[str, Any]:
        """
        Look up a user by object ID or UPN via Graph API.

        Useful for resolving email addresses to Azure AD object IDs when
        building dynamic approval flows.

        Args:
            user_id_or_upn: Object GUID or UPN (e.g. "alice@contoso.com").

        Returns:
            Graph user object dict with id, displayName, mail, etc.
        """
        url = f"{GRAPH_BASE_URL}/users/{user_id_or_upn}"
        return self._request("GET", url, params={"$select": "id,displayName,mail,userPrincipalName"})

    def send_teams_message(
        self,
        team_id: str,
        channel_id: str,
        message_body: str,
        *,
        content_type: str = "text",
    ) -> dict[str, Any]:
        """
        Post a message to a Microsoft Teams channel.

        Args:
            team_id: Teams group object ID.
            channel_id: Channel object ID.
            message_body: Message text (plain text or HTML if content_type="html").
            content_type: "text" or "html".

        Returns:
            Created message object dict.
        """
        url = f"{GRAPH_BASE_URL}/teams/{team_id}/channels/{channel_id}/messages"
        payload = {"body": {"contentType": content_type, "content": message_body}}
        return self._request("POST", url, json_body=payload)


# ============================================================
# QUICK START — run as a script to verify your credentials
# ============================================================

if __name__ == "__main__":
    import json

    client = GraphApiClient.from_client_credentials(
        tenant_id=AZURE_TENANT_ID,
        client_id=AZURE_CLIENT_ID,
        client_secret=AZURE_CLIENT_SECRET,
    )

    environment_id = DEFAULT_ENVIRONMENT_ID
    print(f"\nListing flows in environment: {environment_id}\n")

    flows = client.list_flows(environment_id)
    for flow in flows[:10]:
        print(f"  {flow.display_name:50s}  state={flow.state}  trigger={flow.trigger_type}")

    if flows:
        print(f"\nFetching recent runs for: {flows[0].display_name}")
        runs = client.get_flow_runs(environment_id, flows[0].flow_id, top=5)
        for run in runs:
            print(f"  {run.run_id}  status={run.status}  started={run.start_time}")
