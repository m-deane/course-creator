"""
Module 01 Exercise: Construct and Validate a Flow Definition

## Overview

Power Automate stores every flow as a JSON definition. When you build a flow
in the designer, the platform serialises your canvas into this JSON format behind
the scenes. Understanding that structure helps you:

  - Read and interpret flow definitions exported from an environment
  - Automate flow creation and modification via the Power Automate Management API
  - Write validation logic that checks flows before they are imported

This exercise asks you to construct a valid flow definition JSON for the
"Daily Weather Email" flow you built in Guide 01, and to write validation
functions that check the definition meets Power Automate's structural requirements.

## How to Work Through This File

1. Read the comments and docstrings for each function.
2. Replace each `raise NotImplementedError(...)` with a working implementation.
3. Run the file (`python 01_first_flow_exercise.py`) — all tests must print PASSED.
4. Every test prints a clear explanation if it fails.

## No Mocks — Real Structure

The JSON structure used here matches the actual Power Automate flow definition
schema. You can export a real flow definition from:
  Power Automate portal → My flows → select flow → ... → Export → Package (.zip)
  (The .zip contains a JSON file with this exact structure.)
"""

import json
from typing import Any


# ---------------------------------------------------------------------------
# PART 1: Flow Definition Constants
#
# Power Automate flow definitions use these fixed string identifiers for the
# trigger and connector types used in the Daily Weather Email flow.
# ---------------------------------------------------------------------------

# Trigger type for scheduled flows
RECURRENCE_TRIGGER_TYPE = "Recurrence"

# Connector identifiers (these are the internal API names)
MSN_WEATHER_CONNECTOR = "shared_msnweather"
OUTLOOK_CONNECTOR = "shared_office365"

# Action type for the weather and email operations
OPEN_API_CONNECTION_ACTION_TYPE = "OpenApiConnection"

# Required top-level keys in any flow definition
REQUIRED_FLOW_KEYS = {"definition", "connectionReferences"}

# Required keys inside the "definition" object
REQUIRED_DEFINITION_KEYS = {"triggers", "actions", "$schema"}

# The Power Automate flow schema version string
FLOW_SCHEMA = "https://schema.management.azure.com/providers/Microsoft.Logic/schemas/2016-06-01/workflowdefinition.json#"


# ---------------------------------------------------------------------------
# PART 2: Build the Flow Definition
# ---------------------------------------------------------------------------


def build_recurrence_trigger(interval: int, frequency: str, hours: list[int]) -> dict:
    """
    Build the trigger section of a scheduled cloud flow definition.

    The trigger tells Power Automate when to start the flow. For a scheduled
    flow the trigger type is "Recurrence" with an interval, a frequency unit,
    and a list of hours within the day to run.

    Parameters
    ----------
    interval : int
        How many frequency units between runs. For "every 1 day", use 1.
    frequency : str
        The unit of the interval. Must be one of:
        "Second", "Minute", "Hour", "Day", "Week", "Month"
    hours : list[int]
        Which hours of the day (0–23) to run. For 8 AM use [8].
        Pass an empty list to use the default (midnight).

    Returns
    -------
    dict
        A dict with a single key "Recurrence" whose value is the trigger
        configuration. Structure:

        {
            "Recurrence": {
                "type": "Recurrence",
                "recurrence": {
                    "interval": <interval>,
                    "frequency": <frequency>,
                    "schedule": {
                        "hours": [<hour>, ...]
                    }
                }
            }
        }

    Example
    -------
    >>> trigger = build_recurrence_trigger(1, "Day", [8])
    >>> trigger["Recurrence"]["type"]
    'Recurrence'
    >>> trigger["Recurrence"]["recurrence"]["interval"]
    1
    """
    raise NotImplementedError(
        "Implement build_recurrence_trigger(). "
        "Return a dict with key 'Recurrence' containing 'type' and 'recurrence' sub-keys."
    )


def build_get_weather_action(location: str, units: str) -> dict:
    """
    Build the action definition for the MSN Weather 'Get current weather' step.

    In Power Automate's JSON, each action that calls a connector uses
    type "OpenApiConnection" and specifies the connector via the
    'host' sub-object and the operation via 'operationId'.

    Parameters
    ----------
    location : str
        The city and country to fetch weather for, e.g. "London, UK".
    units : str
        The unit system. Must be either "Imperial" or "Metric".

    Returns
    -------
    dict
        A dict with a single key "Get_current_weather" whose value is:

        {
            "type": "OpenApiConnection",
            "inputs": {
                "host": {
                    "connectionName": "shared_msnweather",
                    "operationId": "CurrentWeather",
                    "apiId": "/providers/Microsoft.PowerApps/apis/shared_msnweather"
                },
                "parameters": {
                    "Location": <location>,
                    "units": <units>
                }
            },
            "runAfter": {}
        }

    Example
    -------
    >>> action = build_get_weather_action("London, UK", "Imperial")
    >>> action["Get_current_weather"]["type"]
    'OpenApiConnection'
    >>> action["Get_current_weather"]["inputs"]["parameters"]["Location"]
    'London, UK'
    """
    raise NotImplementedError(
        "Implement build_get_weather_action(). "
        "Return a dict with key 'Get_current_weather' containing type, inputs, and runAfter."
    )


def build_send_email_action(to_address: str, subject_expression: str, body_expression: str) -> dict:
    """
    Build the action definition for the Office 365 Outlook 'Send an email (V2)' step.

    This action runs after 'Get_current_weather' succeeds, so its 'runAfter'
    section references that action name with the expected status 'Succeeded'.

    Parameters
    ----------
    to_address : str
        The recipient email address.
    subject_expression : str
        The email subject. May contain Power Automate expressions using
        @{...} syntax, e.g. "Today's Weather: @{body('Get_current_weather')?['responses/Summary']}"
    body_expression : str
        The email body. Same expression syntax applies.

    Returns
    -------
    dict
        A dict with a single key "Send_an_email" whose value is:

        {
            "type": "OpenApiConnection",
            "inputs": {
                "host": {
                    "connectionName": "shared_office365",
                    "operationId": "SendEmailV2",
                    "apiId": "/providers/Microsoft.PowerApps/apis/shared_office365"
                },
                "parameters": {
                    "emailMessage/To": <to_address>,
                    "emailMessage/Subject": <subject_expression>,
                    "emailMessage/Body": <body_expression>
                }
            },
            "runAfter": {
                "Get_current_weather": ["Succeeded"]
            }
        }

    Example
    -------
    >>> action = build_send_email_action("alice@example.com", "Weather", "Body text")
    >>> action["Send_an_email"]["inputs"]["parameters"]["emailMessage/To"]
    'alice@example.com'
    >>> action["Send_an_email"]["runAfter"]["Get_current_weather"]
    ['Succeeded']
    """
    raise NotImplementedError(
        "Implement build_send_email_action(). "
        "Return a dict with key 'Send_an_email'. The runAfter must reference 'Get_current_weather'."
    )


def build_connection_references() -> dict:
    """
    Build the connectionReferences section of the flow definition.

    Connection references link the internal connector names used in actions
    to the actual authenticated connections in the environment. Each entry
    needs a 'connection' sub-object with a 'connectionName'.

    Returns
    -------
    dict
        A dict with two keys, "shared_msnweather" and "shared_office365",
        each containing:

        {
            "connection": {
                "connectionName": <same as the outer key>
            }
        }

    Example
    -------
    >>> refs = build_connection_references()
    >>> "shared_msnweather" in refs
    True
    >>> refs["shared_office365"]["connection"]["connectionName"]
    'shared_office365'
    """
    raise NotImplementedError(
        "Implement build_connection_references(). "
        "Return a dict with entries for both MSN Weather and Office 365 Outlook connectors."
    )


def build_daily_weather_flow_definition(
    location: str,
    units: str,
    to_address: str,
    run_hour: int = 8
) -> dict:
    """
    Assemble the complete flow definition JSON for the Daily Weather Email flow.

    This is the top-level function that combines all the pieces: triggers,
    actions, schema, and connection references.

    Parameters
    ----------
    location : str
        City and country for the weather lookup, e.g. "London, UK".
    units : str
        "Imperial" or "Metric".
    to_address : str
        Recipient email address for the daily weather summary.
    run_hour : int
        Hour of the day (0–23) to run the flow. Default is 8 (8 AM).

    Returns
    -------
    dict
        The complete flow definition. Top-level structure:

        {
            "definition": {
                "$schema": <FLOW_SCHEMA>,
                "triggers": { ... },
                "actions": { ... }
            },
            "connectionReferences": { ... }
        }

    Example
    -------
    >>> flow = build_daily_weather_flow_definition("Paris, France", "Metric", "bob@example.com")
    >>> "definition" in flow
    True
    >>> "connectionReferences" in flow
    True
    >>> "triggers" in flow["definition"]
    True
    >>> "actions" in flow["definition"]
    True
    """
    # Build the subject and body expressions using Power Automate's expression syntax.
    # @{body('Get_current_weather')?['responses/Summary']} is how a downstream
    # action references the 'Summary' output of the Get_current_weather action.
    subject_expression = (
        "Today's Weather: "
        "@{body('Get_current_weather')?['responses/Summary']}"
    )
    body_expression = (
        "Good morning!\n\n"
        f"Today's weather for {location}:\n\n"
        "Conditions:   @{body('Get_current_weather')?['responses/Summary']}\n"
        "Temperature:  @{body('Get_current_weather')?['responses/Temperature']} degrees\n"
        "Feels Like:   @{body('Get_current_weather')?['responses/FeelsLike']} degrees\n"
        "Humidity:     @{body('Get_current_weather')?['responses/Humidity']}%\n"
        "Wind Speed:   @{body('Get_current_weather')?['responses/WindSpeed']}\n\n"
        "Have a great day!"
    )

    raise NotImplementedError(
        "Implement build_daily_weather_flow_definition(). "
        "Use build_recurrence_trigger(), build_get_weather_action(), "
        "build_send_email_action(), and build_connection_references() to assemble "
        "and return the complete flow definition dict."
    )


# ---------------------------------------------------------------------------
# PART 3: Validation Functions
# ---------------------------------------------------------------------------


def validate_flow_definition(flow_def: dict) -> tuple[bool, list[str]]:
    """
    Validate that a flow definition dict meets Power Automate's structural requirements.

    This function performs structural validation only — it does not make any
    network calls or check authentication. It checks that all required keys are
    present and that the basic shape is correct.

    Parameters
    ----------
    flow_def : dict
        The flow definition to validate (as returned by build_daily_weather_flow_definition).

    Returns
    -------
    tuple[bool, list[str]]
        (True, []) if the definition is structurally valid.
        (False, [list of error strings]) if validation fails.
        Each error string explains what is missing or wrong.

    Validation Rules
    ----------------
    The function must enforce all of the following:

    1. `flow_def` is a dict.
    2. `flow_def` contains all keys in REQUIRED_FLOW_KEYS ("definition", "connectionReferences").
    3. `flow_def["definition"]` is a dict containing all keys in REQUIRED_DEFINITION_KEYS
       ("triggers", "actions", "$schema").
    4. `flow_def["definition"]["$schema"]` equals FLOW_SCHEMA.
    5. `flow_def["definition"]["triggers"]` is a non-empty dict.
    6. `flow_def["definition"]["actions"]` is a non-empty dict.
    7. `flow_def["connectionReferences"]` is a non-empty dict.
    8. At least one trigger in "triggers" has type equal to RECURRENCE_TRIGGER_TYPE or
       the string "Request" (for HTTP-triggered flows). Accept either.
    9. At least one action in "actions" has type equal to OPEN_API_CONNECTION_ACTION_TYPE.

    Example
    -------
    >>> flow = build_daily_weather_flow_definition("London, UK", "Imperial", "a@b.com")
    >>> valid, errors = validate_flow_definition(flow)
    >>> valid
    True
    >>> errors
    []
    """
    raise NotImplementedError(
        "Implement validate_flow_definition(). "
        "Return (True, []) for a valid definition or (False, [error1, error2, ...]) "
        "listing every rule that failed."
    )


def validate_action_runafter(actions: dict) -> tuple[bool, list[str]]:
    """
    Validate that every action's 'runAfter' references an action that exists in the dict.

    A broken runAfter reference (pointing to a non-existent action name) is one of
    the most common import errors when sharing flow definitions between environments.

    Parameters
    ----------
    actions : dict
        The "actions" section of a flow definition (flow_def["definition"]["actions"]).

    Returns
    -------
    tuple[bool, list[str]]
        (True, []) if all runAfter references are satisfied.
        (False, [error strings]) listing each dangling reference.

    Validation Rule
    ---------------
    For every action A in `actions`:
      For every action name N referenced in A["runAfter"]:
        N must be a key in `actions`.

    Actions with an empty runAfter ({}) are valid — they run first in parallel.

    Example
    -------
    >>> actions = {
    ...     "Step_A": {"type": "OpenApiConnection", "inputs": {}, "runAfter": {}},
    ...     "Step_B": {"type": "OpenApiConnection", "inputs": {}, "runAfter": {"Step_A": ["Succeeded"]}},
    ... }
    >>> valid, errors = validate_action_runafter(actions)
    >>> valid
    True

    >>> actions_broken = {
    ...     "Step_B": {"type": "OpenApiConnection", "inputs": {}, "runAfter": {"Step_A": ["Succeeded"]}},
    ... }
    >>> valid, errors = validate_action_runafter(actions_broken)
    >>> valid
    False
    >>> "Step_A" in errors[0]
    True
    """
    raise NotImplementedError(
        "Implement validate_action_runafter(). "
        "Check that every name referenced in any action's runAfter exists as a key in actions."
    )


def extract_connector_names(flow_def: dict) -> list[str]:
    """
    Extract a sorted list of all connector names used in the flow's actions.

    This is useful for auditing which connectors a flow depends on before
    importing it into an environment where those connectors may not be available.

    Parameters
    ----------
    flow_def : dict
        The complete flow definition dict.

    Returns
    -------
    list[str]
        Sorted list of unique connector names (the "connectionName" values from
        each action's inputs.host.connectionName field). Only actions of type
        "OpenApiConnection" have this field; skip other action types silently.

    Example
    -------
    >>> flow = build_daily_weather_flow_definition("London, UK", "Imperial", "a@b.com")
    >>> connectors = extract_connector_names(flow)
    >>> "shared_msnweather" in connectors
    True
    >>> "shared_office365" in connectors
    True
    >>> connectors == sorted(set(connectors))   # sorted and deduplicated
    True
    """
    raise NotImplementedError(
        "Implement extract_connector_names(). "
        "Iterate over flow_def['definition']['actions'], collect connectionName from "
        "each OpenApiConnection action's inputs.host, and return sorted unique names."
    )


# ---------------------------------------------------------------------------
# PART 4: Tests
# ---------------------------------------------------------------------------

def _assert(condition: bool, message: str) -> None:
    """Raise AssertionError with a descriptive message if condition is False."""
    if not condition:
        raise AssertionError(f"FAILED: {message}")


def test_build_recurrence_trigger() -> None:
    """Tests for build_recurrence_trigger()."""
    trigger = build_recurrence_trigger(1, "Day", [8])

    _assert(isinstance(trigger, dict),
            "build_recurrence_trigger() must return a dict.")
    _assert("Recurrence" in trigger,
            "Return dict must have key 'Recurrence'.")

    rec = trigger["Recurrence"]
    _assert(rec.get("type") == RECURRENCE_TRIGGER_TYPE,
            f"trigger['Recurrence']['type'] must be '{RECURRENCE_TRIGGER_TYPE}', got {rec.get('type')!r}.")
    _assert("recurrence" in rec,
            "trigger['Recurrence'] must contain key 'recurrence'.")

    recurrence = rec["recurrence"]
    _assert(recurrence.get("interval") == 1,
            f"recurrence['interval'] must be 1, got {recurrence.get('interval')!r}.")
    _assert(recurrence.get("frequency") == "Day",
            f"recurrence['frequency'] must be 'Day', got {recurrence.get('frequency')!r}.")
    _assert("schedule" in recurrence,
            "recurrence must contain key 'schedule'.")
    _assert(recurrence["schedule"].get("hours") == [8],
            f"recurrence['schedule']['hours'] must be [8], got {recurrence['schedule'].get('hours')!r}.")

    # Empty hours list is also valid (midnight default)
    trigger_midnight = build_recurrence_trigger(1, "Day", [])
    _assert(trigger_midnight["Recurrence"]["recurrence"]["schedule"]["hours"] == [],
            "Empty hours list must be preserved, not converted to something else.")

    print("PASSED: test_build_recurrence_trigger")


def test_build_get_weather_action() -> None:
    """Tests for build_get_weather_action()."""
    action = build_get_weather_action("London, UK", "Imperial")

    _assert(isinstance(action, dict),
            "build_get_weather_action() must return a dict.")
    _assert("Get_current_weather" in action,
            "Return dict must have key 'Get_current_weather'.")

    step = action["Get_current_weather"]
    _assert(step.get("type") == OPEN_API_CONNECTION_ACTION_TYPE,
            f"type must be '{OPEN_API_CONNECTION_ACTION_TYPE}', got {step.get('type')!r}.")
    _assert("inputs" in step,
            "'Get_current_weather' must have an 'inputs' key.")
    _assert("host" in step["inputs"],
            "inputs must contain 'host'.")
    _assert(step["inputs"]["host"].get("connectionName") == MSN_WEATHER_CONNECTOR,
            f"host.connectionName must be '{MSN_WEATHER_CONNECTOR}'.")
    _assert(step["inputs"]["host"].get("operationId") == "CurrentWeather",
            "host.operationId must be 'CurrentWeather'.")
    _assert("parameters" in step["inputs"],
            "inputs must contain 'parameters'.")
    _assert(step["inputs"]["parameters"].get("Location") == "London, UK",
            "parameters['Location'] must equal the location argument.")
    _assert(step["inputs"]["parameters"].get("units") == "Imperial",
            "parameters['units'] must equal the units argument.")
    _assert("runAfter" in step,
            "'Get_current_weather' must have a 'runAfter' key.")
    _assert(step["runAfter"] == {},
            "runAfter for the first action must be an empty dict (it runs first).")

    print("PASSED: test_build_get_weather_action")


def test_build_send_email_action() -> None:
    """Tests for build_send_email_action()."""
    action = build_send_email_action("alice@example.com", "Subject text", "Body text")

    _assert(isinstance(action, dict),
            "build_send_email_action() must return a dict.")
    _assert("Send_an_email" in action,
            "Return dict must have key 'Send_an_email'.")

    step = action["Send_an_email"]
    _assert(step.get("type") == OPEN_API_CONNECTION_ACTION_TYPE,
            f"type must be '{OPEN_API_CONNECTION_ACTION_TYPE}', got {step.get('type')!r}.")
    _assert("inputs" in step,
            "'Send_an_email' must have an 'inputs' key.")
    _assert(step["inputs"]["host"].get("connectionName") == OUTLOOK_CONNECTOR,
            f"host.connectionName must be '{OUTLOOK_CONNECTOR}'.")
    _assert(step["inputs"]["host"].get("operationId") == "SendEmailV2",
            "host.operationId must be 'SendEmailV2'.")
    params = step["inputs"]["parameters"]
    _assert(params.get("emailMessage/To") == "alice@example.com",
            "parameters['emailMessage/To'] must equal the to_address argument.")
    _assert(params.get("emailMessage/Subject") == "Subject text",
            "parameters['emailMessage/Subject'] must equal the subject_expression argument.")
    _assert(params.get("emailMessage/Body") == "Body text",
            "parameters['emailMessage/Body'] must equal the body_expression argument.")

    run_after = step.get("runAfter", {})
    _assert("Get_current_weather" in run_after,
            "runAfter must reference 'Get_current_weather'.")
    _assert(run_after["Get_current_weather"] == ["Succeeded"],
            "runAfter['Get_current_weather'] must be ['Succeeded'].")

    print("PASSED: test_build_send_email_action")


def test_build_connection_references() -> None:
    """Tests for build_connection_references()."""
    refs = build_connection_references()

    _assert(isinstance(refs, dict),
            "build_connection_references() must return a dict.")
    _assert(MSN_WEATHER_CONNECTOR in refs,
            f"connectionReferences must contain key '{MSN_WEATHER_CONNECTOR}'.")
    _assert(OUTLOOK_CONNECTOR in refs,
            f"connectionReferences must contain key '{OUTLOOK_CONNECTOR}'.")
    _assert(refs[MSN_WEATHER_CONNECTOR]["connection"]["connectionName"] == MSN_WEATHER_CONNECTOR,
            "MSN Weather connection.connectionName must match the connector key.")
    _assert(refs[OUTLOOK_CONNECTOR]["connection"]["connectionName"] == OUTLOOK_CONNECTOR,
            "Outlook connection.connectionName must match the connector key.")

    print("PASSED: test_build_connection_references")


def test_build_daily_weather_flow_definition() -> None:
    """Tests for build_daily_weather_flow_definition()."""
    flow = build_daily_weather_flow_definition(
        location="Tokyo, Japan",
        units="Metric",
        to_address="user@example.com",
        run_hour=7
    )

    _assert(isinstance(flow, dict),
            "build_daily_weather_flow_definition() must return a dict.")
    _assert("definition" in flow,
            "Flow must have top-level key 'definition'.")
    _assert("connectionReferences" in flow,
            "Flow must have top-level key 'connectionReferences'.")

    defn = flow["definition"]
    _assert("$schema" in defn,
            "definition must contain '$schema'.")
    _assert(defn["$schema"] == FLOW_SCHEMA,
            f"$schema must equal FLOW_SCHEMA constant.")
    _assert("triggers" in defn,
            "definition must contain 'triggers'.")
    _assert("actions" in defn,
            "definition must contain 'actions'.")
    _assert("Recurrence" in defn["triggers"],
            "triggers must contain a 'Recurrence' key.")
    _assert("Get_current_weather" in defn["actions"],
            "actions must contain 'Get_current_weather'.")
    _assert("Send_an_email" in defn["actions"],
            "actions must contain 'Send_an_email'.")

    # Verify run_hour is wired through
    recurrence = defn["triggers"]["Recurrence"]["recurrence"]
    _assert(7 in recurrence["schedule"]["hours"],
            "run_hour=7 must appear in the trigger's schedule.hours list.")

    # Verify location is in the weather action
    weather_params = defn["actions"]["Get_current_weather"]["inputs"]["parameters"]
    _assert(weather_params.get("Location") == "Tokyo, Japan",
            "Weather action parameters['Location'] must reflect the location argument.")
    _assert(weather_params.get("units") == "Metric",
            "Weather action parameters['units'] must reflect the units argument.")

    # Verify to_address is in the email action
    email_params = defn["actions"]["Send_an_email"]["inputs"]["parameters"]
    _assert(email_params.get("emailMessage/To") == "user@example.com",
            "Email action parameters['emailMessage/To'] must reflect the to_address argument.")

    print("PASSED: test_build_daily_weather_flow_definition")


def test_validate_flow_definition() -> None:
    """Tests for validate_flow_definition()."""
    # A valid definition passes
    flow = build_daily_weather_flow_definition("London, UK", "Imperial", "a@b.com")
    valid, errors = validate_flow_definition(flow)
    _assert(valid is True,
            f"A correctly built flow definition must pass validation. Errors: {errors}")
    _assert(errors == [],
            f"Error list must be empty for a valid definition. Got: {errors}")

    # Not a dict
    valid, errors = validate_flow_definition("not a dict")
    _assert(valid is False,
            "A non-dict input must fail validation.")
    _assert(len(errors) > 0,
            "Error list must not be empty when validation fails.")

    # Missing 'connectionReferences'
    broken = {"definition": flow["definition"]}
    valid, errors = validate_flow_definition(broken)
    _assert(valid is False,
            "Missing 'connectionReferences' must fail validation.")
    _assert(any("connectionReferences" in e for e in errors),
            "Error message must mention the missing 'connectionReferences' key.")

    # Wrong $schema
    import copy
    bad_schema = copy.deepcopy(flow)
    bad_schema["definition"]["$schema"] = "http://wrong-schema.example.com"
    valid, errors = validate_flow_definition(bad_schema)
    _assert(valid is False,
            "A wrong $schema value must fail validation.")

    # Empty triggers
    no_triggers = copy.deepcopy(flow)
    no_triggers["definition"]["triggers"] = {}
    valid, errors = validate_flow_definition(no_triggers)
    _assert(valid is False,
            "Empty triggers dict must fail validation.")

    print("PASSED: test_validate_flow_definition")


def test_validate_action_runafter() -> None:
    """Tests for validate_action_runafter()."""
    # Valid: B runs after A, A runs first
    actions_valid = {
        "Step_A": {"type": OPEN_API_CONNECTION_ACTION_TYPE, "inputs": {}, "runAfter": {}},
        "Step_B": {"type": OPEN_API_CONNECTION_ACTION_TYPE, "inputs": {}, "runAfter": {"Step_A": ["Succeeded"]}},
    }
    valid, errors = validate_action_runafter(actions_valid)
    _assert(valid is True,
            f"Valid runAfter references must pass. Errors: {errors}")
    _assert(errors == [],
            f"Error list must be empty for valid runAfter. Got: {errors}")

    # Invalid: B references Step_A which does not exist
    actions_broken = {
        "Step_B": {"type": OPEN_API_CONNECTION_ACTION_TYPE, "inputs": {}, "runAfter": {"Step_A": ["Succeeded"]}},
    }
    valid, errors = validate_action_runafter(actions_broken)
    _assert(valid is False,
            "Dangling runAfter reference must fail validation.")
    _assert(any("Step_A" in e for e in errors),
            "Error message must mention the missing action name 'Step_A'.")

    # Empty dict: no actions, trivially valid
    valid, errors = validate_action_runafter({})
    _assert(valid is True,
            "Empty actions dict must be trivially valid.")

    print("PASSED: test_validate_action_runafter")


def test_extract_connector_names() -> None:
    """Tests for extract_connector_names()."""
    flow = build_daily_weather_flow_definition("London, UK", "Imperial", "a@b.com")
    connectors = extract_connector_names(flow)

    _assert(isinstance(connectors, list),
            "extract_connector_names() must return a list.")
    _assert(MSN_WEATHER_CONNECTOR in connectors,
            f"'{MSN_WEATHER_CONNECTOR}' must be in the connector list.")
    _assert(OUTLOOK_CONNECTOR in connectors,
            f"'{OUTLOOK_CONNECTOR}' must be in the connector list.")
    _assert(connectors == sorted(set(connectors)),
            "Connector list must be sorted and deduplicated.")

    # If a non-OpenApiConnection action exists, it must not add anything
    flow_with_extra = {
        "definition": {
            "$schema": FLOW_SCHEMA,
            "triggers": flow["definition"]["triggers"],
            "actions": {
                **flow["definition"]["actions"],
                "Compose_Step": {
                    "type": "Compose",  # not OpenApiConnection — no connectionName
                    "inputs": "some value",
                    "runAfter": {}
                }
            }
        },
        "connectionReferences": flow["connectionReferences"]
    }
    connectors_extra = extract_connector_names(flow_with_extra)
    # Should still only list the two real connectors, not crash on the Compose action
    _assert(len(connectors_extra) == 2,
            f"Only OpenApiConnection actions contribute connector names. "
            f"Expected 2, got {len(connectors_extra)}: {connectors_extra}")

    print("PASSED: test_extract_connector_names")


# ---------------------------------------------------------------------------
# PART 5: Run All Tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_build_recurrence_trigger,
        test_build_get_weather_action,
        test_build_send_email_action,
        test_build_connection_references,
        test_build_daily_weather_flow_definition,
        test_validate_flow_definition,
        test_validate_action_runafter,
        test_extract_connector_names,
    ]

    passed = 0
    failed = 0

    for test_fn in tests:
        try:
            test_fn()
            passed += 1
        except NotImplementedError as exc:
            print(f"NOT IMPLEMENTED: {test_fn.__name__} — {exc}")
            failed += 1
        except AssertionError as exc:
            print(f"{exc}")
            failed += 1
        except Exception as exc:
            print(f"ERROR in {test_fn.__name__}: {type(exc).__name__}: {exc}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    if failed == 0:
        print("All tests passed. Flow definition structure is correct.")
    else:
        print(f"{failed} test(s) still need implementation or fixes.")
    print("=" * 60)
