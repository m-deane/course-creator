# Module 0 Quiz: LLM Mesh Foundations

**Course:** Gen AI & Dataiku: LLM Mesh Use Cases
**Module:** 0 - LLM Mesh Foundations
**Time Limit:** 20 minutes
**Total Points:** 100
**Passing Score:** 70/100

## Instructions

- This quiz has 12 questions covering LLM Mesh architecture, provider setup, and governance
- Select the best answer for each question
- Point values are indicated for each question
- You have 2 attempts per question
- Refer to module guides and notebooks if needed

---

## Section 1: LLM Mesh Architecture (30 points)

### Question 1 (8 points)

What is the PRIMARY purpose of Dataiku's LLM Mesh abstraction layer?

A) To reduce the cost of LLM API calls by caching responses
B) To provide a single unified API for multiple LLM providers
C) To automatically select the cheapest LLM provider for each request
D) To host LLM models directly within Dataiku DSS

**Answer:** B

**Explanation:**
- **B (Correct):** LLM Mesh provides provider abstraction, allowing developers to use a single API to interact with multiple LLM providers (Anthropic, OpenAI, Azure, etc.) without changing code.
- **A:** While caching may be a feature, the primary purpose is provider abstraction and unified access.
- **C:** LLM Mesh doesn't automatically route by cost; routing and selection are configured by administrators.
- **D:** LLM Mesh connects to external providers; it doesn't host models locally.

---

### Question 2 (8 points)

In the LLM Mesh architecture, where are LLM connections configured?

A) At the project level, unique to each DSS project
B) At the instance level, available to all authorized projects
C) In each Python recipe that uses LLM functionality
D) In individual user profile settings

**Answer:** B

**Explanation:**
- **B (Correct):** LLM Mesh connections are configured at the DSS instance level by administrators, making them available to all projects with appropriate permissions.
- **A:** Connections are not project-specific; they're centralized for governance.
- **C:** Python recipes reference existing connections; they don't define them.
- **D:** Connections are not user-specific; they're shared resources with role-based access control.

---

### Question 3 (7 points)

Which of the following is NOT a key benefit of using LLM Mesh?

A) Centralized cost tracking across all LLM usage
B) Automatic prompt optimization for better results
C) Role-based access control to LLM connections
D) Provider failover and redundancy options

**Answer:** B

**Explanation:**
- **B (Correct):** LLM Mesh does not automatically optimize prompts; that's the user's responsibility using tools like Prompt Studios.
- **A:** Cost tracking is a core governance feature of LLM Mesh.
- **C:** Role-based access control is essential for enterprise governance.
- **D:** LLM Mesh supports failover configurations for reliability.

---

### Question 4 (7 points)

What happens when a project tries to use an LLM connection that the user doesn't have access to?

A) The request is automatically routed to a default public LLM
B) The system prompts the user to enter an API key
C) An authentication error is returned and the request fails
D) The request is queued for administrator approval

**Answer:** C

**Explanation:**
- **C (Correct):** LLM Mesh enforces access control; unauthorized access attempts result in authentication errors.
- **A:** There's no automatic fallback to public LLMs; this would bypass governance.
- **B:** API keys are configured at the instance level, not entered by users.
- **D:** Requests aren't queued; access must be granted before making requests.

---

## Section 2: Provider Configuration (35 points)

### Question 5 (9 points)

When configuring an Anthropic Claude connection in LLM Mesh, which of the following is required?

A) A Dataiku-hosted API gateway URL
B) An Anthropic API key with appropriate permissions
C) A virtual private cloud (VPC) endpoint
D) Multi-factor authentication credentials

**Answer:** B

**Explanation:**
- **B (Correct):** An Anthropic API key is required to authenticate requests to Anthropic's API.
- **A:** Requests go directly to Anthropic's API, not through a Dataiku gateway.
- **C:** VPC endpoints are optional for enhanced security, not required.
- **D:** MFA is not required for API key configuration.

---

### Question 6 (9 points)

Your organization uses Azure OpenAI. What additional configuration is needed compared to standard OpenAI?

A) Only the API key needs to be changed
B) The deployment name and Azure endpoint URL must be specified
C) A separate Azure Active Directory integration
D) Custom model weights must be uploaded

**Answer:** B

**Explanation:**
- **B (Correct):** Azure OpenAI requires the Azure resource endpoint URL and the specific deployment name, in addition to credentials.
- **A:** Azure OpenAI has different endpoints and requires deployment-specific configuration.
- **C:** While Azure AD can be used for auth, it's not required; API keys work.
- **D:** Azure OpenAI uses Microsoft's hosted models; you don't upload weights.

---

### Question 7 (8 points)

You've configured multiple LLM connections (Claude, GPT-4, GPT-3.5). How does a Python recipe specify which connection to use?

A) By setting environment variable `LLM_PROVIDER`
B) By passing the connection name when creating an LLM handle
C) The system automatically selects based on prompt complexity
D) By importing provider-specific Python libraries

**Answer:** B

**Explanation:**
- **B (Correct):** Use `LLMHandle("connection-name")` to specify which configured connection to use.
- **A:** Connection selection isn't done via environment variables.
- **C:** LLM Mesh doesn't automatically route; explicit selection is required.
- **D:** The LLM Mesh API is unified; you don't import provider-specific libraries.

---

### Question 8 (9 points)

Which statement about LLM Mesh connection testing is TRUE?

A) Connections can only be tested by making actual API calls from a project
B) The LLM Mesh admin interface provides a built-in connection testing feature
C) Test calls don't count toward API usage limits
D) Connection tests automatically validate prompt quality

**Answer:** B

**Explanation:**
- **B (Correct):** The LLM Mesh administration interface includes tools to test connections before deploying them to projects.
- **A:** While project-level testing is possible, admin testing is available and recommended.
- **C:** Test calls use the actual API and count toward usage/costs.
- **D:** Connection tests verify connectivity and authentication, not prompt quality.

---

## Section 3: Governance and Best Practices (35 points)

### Question 9 (9 points)

What is the primary purpose of cost tracking in LLM Mesh?

A) To automatically stop requests when budget limits are exceeded
B) To provide visibility into LLM usage by project, user, and model
C) To negotiate better pricing with LLM providers
D) To convert all costs to a standard currency

**Answer:** B

**Explanation:**
- **B (Correct):** Cost tracking provides transparency and attribution of LLM usage across the organization.
- **A:** Cost tracking is monitoring/reporting; automatic enforcement requires additional configuration.
- **C:** Tracking doesn't directly negotiate pricing with vendors.
- **D:** While currency conversion may occur, the primary purpose is usage visibility.

---

### Question 10 (9 points)

An administrator wants to prevent a specific project from using expensive GPT-4 models. How should this be implemented?

A) Remove the API key from the GPT-4 connection
B) Configure connection-level permissions to exclude that project
C) Set rate limits to zero for that project
D) Use IP filtering to block the project's requests

**Answer:** B

**Explanation:**
- **B (Correct):** LLM Mesh supports granular permissions, allowing administrators to grant or deny project-level access to specific connections.
- **A:** Removing the API key affects all projects, not just one.
- **C:** Rate limits control request frequency, not access permissions.
- **D:** IP filtering is not the appropriate mechanism for project-based access control.

---

### Question 11 (8 points)

Which metric is MOST important for monitoring LLM Mesh health in production?

A) The total number of LLM connections configured
B) Request success rate and error types
C) The average length of user prompts
D) Number of projects with LLM access

**Answer:** B

**Explanation:**
- **B (Correct):** Success rate and error patterns indicate system health and potential issues with connections or quotas.
- **A:** Number of connections is a configuration metric, not a health indicator.
- **C:** Prompt length is a usage characteristic, not a health metric.
- **D:** Project count is a governance metric, not a health indicator.

---

### Question 12 (9 points)

Your organization has compliance requirements to log all LLM interactions. What LLM Mesh feature supports this?

A) Automatic prompt sanitization
B) Audit logging of all LLM requests and responses
C) Encrypted storage of API keys
D) Real-time compliance scanning of outputs

**Answer:** B

**Explanation:**
- **B (Correct):** LLM Mesh provides comprehensive audit logging capabilities for compliance and monitoring.
- **A:** Prompt sanitization is a separate concern; logging captures actual content.
- **C:** Key encryption is about security, not interaction logging.
- **D:** While useful, real-time scanning isn't the primary logging mechanism for compliance.

---

## Answer Key Summary

| Question | Answer | Points | Topic |
|----------|--------|--------|-------|
| 1 | B | 8 | LLM Mesh purpose |
| 2 | B | 8 | Connection configuration |
| 3 | B | 7 | LLM Mesh benefits |
| 4 | C | 7 | Access control |
| 5 | B | 9 | Anthropic setup |
| 6 | B | 9 | Azure OpenAI config |
| 7 | B | 8 | Connection selection |
| 8 | B | 9 | Connection testing |
| 9 | B | 9 | Cost tracking |
| 10 | B | 9 | Access permissions |
| 11 | B | 8 | Monitoring metrics |
| 12 | B | 9 | Audit logging |

**Total:** 100 points

---

## Performance Indicators

**90-100:** Excellent understanding of LLM Mesh fundamentals
**80-89:** Strong grasp with minor gaps
**70-79:** Adequate understanding, review governance concepts
**Below 70:** Review module materials, especially provider configuration and governance

## Next Steps

After completing this quiz:
1. Review any questions you missed
2. Revisit relevant guide sections
3. Complete the module notebooks for hands-on practice
4. Proceed to Module 1: Prompt Design when ready
