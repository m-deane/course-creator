# Module 4 Quiz: Deployment and Governance

**Course:** Gen AI & Dataiku: LLM Mesh Use Cases
**Module:** 4 - Deployment and Governance
**Time Limit:** 20 minutes
**Total Points:** 100
**Passing Score:** 70/100

## Instructions

- This quiz has 14 questions covering API deployment, webapp integration, monitoring, and governance
- Select the best answer for each question
- Point values are indicated for each question
- You have 2 attempts per question
- Refer to module guides and notebooks if needed

---

## Section 1: API Deployment (28 points)

### Question 1 (7 points)

What is the primary purpose of deploying an LLM application as a Dataiku API endpoint?

A) To make the LLM application accessible to external systems and users
B) To automatically optimize the LLM's performance
C) To reduce the cost of LLM API calls
D) To backup LLM responses automatically

**Answer:** A

**Explanation:**
- **A (Correct):** API endpoints expose LLM functionality to other applications, services, and users programmatically.
- **B:** APIs expose functionality; optimization is a separate concern.
- **C:** Deployment doesn't inherently reduce costs; it provides access.
- **D:** Backup is a separate data management function.

---

### Question 2 (7 points)

Which deployment type allows for automatic scaling based on request load?

A) Batch deployment
B) API endpoint deployment
C) Manual deployment
D) Notebook deployment

**Answer:** B

**Explanation:**
- **B (Correct):** Dataiku API endpoints can auto-scale to handle varying request loads.
- **A:** Batch deployments run on schedules, not in response to real-time load.
- **C:** Manual deployment doesn't provide auto-scaling.
- **D:** Notebooks are for interactive development, not production deployment.

---

### Question 3 (7 points)

What authentication mechanism does Dataiku use for API endpoints?

A) No authentication (public access)
B) API keys associated with user accounts
C) OAuth 2.0 only
D) Basic authentication with username/password

**Answer:** B

**Explanation:**
- **B (Correct):** Dataiku uses API keys that are tied to user accounts and inherit their permissions.
- **A:** APIs require authentication for security and governance.
- **C:** While OAuth may be supported, API keys are the primary mechanism.
- **D:** API keys are preferred over basic auth for programmatic access.

---

### Question 4 (7 points)

Your API endpoint needs to handle a prompt template with variables. What's the best approach?

A) Hardcode the prompt in the API and accept only variable values as input
B) Accept the entire prompt as input on each request
C) Reference a Prompt Studio template and accept variables as API parameters
D) Store prompts in environment variables

**Answer:** C

**Explanation:**
- **C (Correct):** Referencing Prompt Studio templates enables centralized management, versioning, and testing while APIs handle variable values.
- **A:** Hardcoding reduces flexibility and requires code changes for prompt updates.
- **B:** Accepting full prompts eliminates governance and versioning benefits.
- **D:** Environment variables aren't appropriate for prompt management.

---

## Section 2: Webapp Integration (28 points)

### Question 5 (7 points)

What is the primary advantage of using Dataiku webapps for LLM applications?

A) Webapps are faster than API endpoints
B) Webapps provide a user interface for non-technical users
C) Webapps don't require any coding
D) Webapps are automatically more secure

**Answer:** B

**Explanation:**
- **B (Correct):** Webapps provide visual interfaces, making LLM applications accessible to users without API integration knowledge.
- **A:** Performance depends on implementation, not deployment type.
- **C:** Webapps typically require some coding (JavaScript, Python backend).
- **D:** Security depends on implementation; webapps aren't inherently more secure.

---

### Question 6 (7 points)

In a Dataiku webapp, which component handles LLM API calls?

A) The frontend HTML/JavaScript only
B) The backend Python code
C) Dataiku handles all LLM calls automatically
D) A separate microservice is always required

**Answer:** B

**Explanation:**
- **B (Correct):** Backend Python code in webapps calls LLM Mesh and returns results to the frontend.
- **A:** Frontend shouldn't make direct API calls (exposes credentials, lacks control).
- **C:** Developers must implement the LLM call logic in their webapp code.
- **D:** Webapps can directly use LLM Mesh; separate services are optional.

---

### Question 7 (7 points)

What is a best practice for handling long-running LLM requests in webapps?

A) Block the UI until the response is ready
B) Use asynchronous processing with loading indicators
C) Set a 1-second timeout and give up if not ready
D) Cache all possible responses in advance

**Answer:** B

**Explanation:**
- **B (Correct):** Async processing with UI feedback provides better user experience for potentially slow LLM calls.
- **A:** Blocking creates poor user experience and may cause timeouts.
- **C:** LLM calls often take longer than 1 second; premature timeouts cause failures.
- **D:** Pre-caching all responses is impossible for dynamic, user-specific queries.

---

### Question 8 (7 points)

How should you handle errors from LLM API calls in a webapp?

A) Silently ignore errors to avoid confusing users
B) Display technical stack traces to the user
C) Show user-friendly error messages and log details for debugging
D) Automatically retry indefinitely until success

**Answer:** C

**Explanation:**
- **C (Correct):** User-facing errors should be friendly while detailed logs help with debugging.
- **A:** Ignoring errors leaves users confused about why nothing happened.
- **B:** Stack traces are too technical for end users and may expose security information.
- **D:** Infinite retries can waste resources; implement retry limits and user notification.

---

## Section 3: Monitoring and Governance (44 points)

### Question 9 (8 points)

Which metrics are MOST important for monitoring production LLM applications?

A) Number of API keys issued
B) Request success rate, latency, and token usage
C) Number of prompt versions created
D) Size of Knowledge Banks in GB

**Answer:** B

**Explanation:**
- **B (Correct):** Success rate indicates reliability, latency measures user experience, and token usage tracks costs.
- **A:** API key count is a governance metric, not operational health.
- **C:** Version count is development-related, not production monitoring.
- **D:** Storage size is infrastructure, not application performance.

---

### Question 10 (8 points)

What is the purpose of tracking token usage per project?

A) To improve LLM response quality
B) To enable cost allocation and budget management
C) To automatically optimize prompts
D) To determine which LLM model is fastest

**Answer:** B

**Explanation:**
- **B (Correct):** Per-project tracking enables chargeback, budget control, and cost awareness.
- **A:** Token tracking is about cost, not quality.
- **C:** Tracking doesn't optimize; it provides data for manual optimization decisions.
- **D:** Token usage measures cost, not speed.

---

### Question 11 (7 points)

Your organization requires audit logs of all LLM interactions. What should be logged?

A) Only the final response text
B) Timestamp, user, prompt, response, model, and token usage
C) Only failed requests for debugging
D) User information only (not prompt content)

**Answer:** B

**Explanation:**
- **B (Correct):** Comprehensive logging includes all relevant context for compliance, debugging, and analysis.
- **A:** Prompts and metadata are essential for understanding interactions.
- **C:** Successful requests are equally important for audit trails.
- **D:** Prompt and response content are often required for compliance.

---

### Question 12 (7 points)

How can you prevent excessive spending on LLM API calls?

A) Set rate limits and budget alerts at the LLM Mesh level
B) Disable all LLM connections at night
C) Only allow administrators to use LLMs
D) Use only free-tier models

**Answer:** A

**Explanation:**
- **A (Correct):** Rate limits and alerts provide proactive cost control while maintaining functionality.
- **B:** Time-based restrictions are too rigid and may impact legitimate use.
- **C:** Overly restrictive access limits LLM utility across the organization.
- **D:** Free tiers have severe limitations and may not meet requirements.

---

### Question 13 (7 points)

What is the purpose of role-based access control (RBAC) for LLM connections?

A) To improve LLM response speed
B) To ensure users only access LLM resources appropriate for their role
C) To automatically assign the cheapest model to each user
D) To prevent LLMs from accessing user data

**Answer:** B

**Explanation:**
- **B (Correct):** RBAC ensures appropriate access based on job function, security clearance, and project needs.
- **A:** RBAC is about authorization, not performance.
- **C:** RBAC controls access, not cost optimization routing.
- **D:** Data access controls are separate from LLM connection access.

---

### Question 14 (7 points)

You notice a spike in error rates for an LLM API. What should you check FIRST?

A) Whether new features have been added to the webapp
B) Provider status, rate limits, and API quota usage
C) The total number of registered users
D) Whether the Knowledge Bank has been updated

**Answer:** B

**Explanation:**
- **B (Correct):** Most API errors stem from provider issues, rate limiting, or quota exhaustion.
- **A:** Application changes are possible causes but external factors should be checked first.
- **C:** User count doesn't directly cause API errors.
- **D:** Knowledge Bank updates don't affect LLM API error rates.

---

## Answer Key Summary

| Question | Answer | Points | Topic |
|----------|--------|--------|-------|
| 1 | A | 7 | API purpose |
| 2 | B | 7 | Auto-scaling |
| 3 | B | 7 | Authentication |
| 4 | C | 7 | Prompt templates |
| 5 | B | 7 | Webapp advantages |
| 6 | B | 7 | Backend processing |
| 7 | B | 7 | Async handling |
| 8 | C | 7 | Error handling |
| 9 | B | 8 | Monitoring metrics |
| 10 | B | 8 | Cost tracking |
| 11 | B | 7 | Audit logging |
| 12 | A | 7 | Cost control |
| 13 | B | 7 | RBAC purpose |
| 14 | B | 7 | Error troubleshooting |

**Total:** 100 points

---

## Performance Indicators

**90-100:** Expert-level understanding of deployment and governance
**80-89:** Strong grasp of production LLM operations
**70-79:** Adequate understanding, review monitoring concepts
**Below 70:** Review module materials, especially governance and monitoring

## Common Mistakes to Avoid

1. **Exposing API keys in frontend**: Always call LLMs from backend code
2. **Blocking UIs**: Use async patterns for LLM calls
3. **Poor error handling**: Show user-friendly messages, log technical details
4. **No cost monitoring**: Track token usage to prevent budget overruns
5. **Insufficient logging**: Log comprehensively for compliance and debugging
6. **Missing rate limits**: Implement limits to prevent runaway costs

## Key Concepts to Master

### API Deployment
- Authentication with API keys
- Auto-scaling for variable load
- Integration with Prompt Studios
- Request/response patterns

### Webapp Development
- Backend (Python) handles LLM calls
- Frontend provides user interface
- Async processing for better UX
- Comprehensive error handling

### Monitoring
- Success rate and error rates
- Request latency
- Token usage and costs
- User activity patterns

### Governance
- Role-based access control
- Audit logging
- Cost allocation and budgets
- Rate limiting and quotas

## Production Checklist

Before deploying to production:
- [ ] Authentication configured
- [ ] Error handling implemented
- [ ] Monitoring and alerts set up
- [ ] Rate limits configured
- [ ] Audit logging enabled
- [ ] Cost tracking active
- [ ] Documentation complete
- [ ] Security review passed
- [ ] Load testing completed
- [ ] Rollback plan defined

## Next Steps

After completing this quiz:
1. Review any questions you missed
2. Complete the deployment hands-on exercises
3. Set up monitoring for a sample application
4. Configure governance policies in your environment
5. Review the entire course and prepare for the capstone project

## Course Completion

Congratulations on completing all module quizzes! You now have comprehensive knowledge of:
- LLM Mesh architecture and configuration
- Prompt design with Prompt Studios
- RAG applications with Knowledge Banks
- Custom Python integration
- Production deployment and governance

**Recommended:** Complete the capstone project to demonstrate your skills in building an end-to-end Gen AI application with Dataiku.
