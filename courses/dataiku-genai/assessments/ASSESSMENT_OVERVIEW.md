# Assessment Overview: Dataiku GenAI Course

## Summary

This document provides an overview of all assessment materials for the **Gen AI & Dataiku: LLM Mesh Use Cases** course.

## Module Quizzes

### Module 0: LLM Mesh Foundations
**Location:** `modules/module_00_llm_mesh/assessments/quiz_module_00.md`
- **Questions:** 12
- **Total Points:** 100
- **Time Limit:** 20 minutes
- **Passing Score:** 70/100

**Topics Covered:**
- LLM Mesh architecture and purpose
- Connection configuration (Anthropic, OpenAI, Azure)
- Provider abstraction and unified API
- Cost tracking and governance
- Access control and permissions
- Audit logging and monitoring

**Key Question Types:**
- Conceptual understanding (30%)
- Provider configuration (35%)
- Governance and best practices (35%)

---

### Module 1: Prompt Design with Prompt Studios
**Location:** `modules/module_01_prompts/assessments/quiz_module_01.md`
- **Questions:** 13 (10 bonus points available)
- **Total Points:** 110 (100 base + 10 bonus)
- **Time Limit:** 20 minutes
- **Passing Score:** 70/100

**Topics Covered:**
- Prompt Studios interface and features
- System vs. user prompts
- Template variables (Mustache syntax)
- Iteration and conditional logic
- Version control and testing
- Few-shot examples
- Production debugging

**Key Question Types:**
- Prompt Studios basics (30%)
- Template variables (35%)
- Testing and optimization (35%)

---

### Module 2: RAG with Knowledge Banks
**Location:** `modules/module_02_rag/assessments/quiz_module_02.md`
- **Questions:** 14 (18 bonus points available)
- **Total Points:** 118 (100 base + 18 bonus)
- **Time Limit:** 20 minutes
- **Passing Score:** 70/100

**Topics Covered:**
- Knowledge Bank purpose and architecture
- Document ingestion and chunking
- Embedding models and vector search
- RAG pipeline flow
- Source attribution
- Chunk size and overlap optimization
- Top-k retrieval configuration
- Knowledge Bank maintenance

**Key Question Types:**
- Knowledge Banks fundamentals (28%)
- RAG architecture (35%)
- Configuration and optimization (37%)

---

### Module 3: Custom LLM Applications
**Location:** `modules/module_03_custom/assessments/quiz_module_03.md`
- **Questions:** 13 (10 bonus points available)
- **Total Points:** 110 (100 base + 10 bonus)
- **Time Limit:** 20 minutes
- **Passing Score:** 70/100

**Topics Covered:**
- Python LLM Mesh API (`dataiku.LLMHandle`)
- Message structure and parameters
- Token usage tracking
- Custom model patterns (wrapper, chain, router, ensemble)
- Pipeline integration strategies
- Error handling and rate limiting
- Production best practices

**Key Question Types:**
- Python LLM Mesh API (32%)
- Custom model patterns (35%)
- Pipeline integration (33%)

---

### Module 4: Deployment and Governance
**Location:** `modules/module_04_deployment/assessments/quiz_module_04.md`
- **Questions:** 14
- **Total Points:** 100
- **Time Limit:** 20 minutes
- **Passing Score:** 70/100

**Topics Covered:**
- API endpoint deployment
- Auto-scaling and authentication
- Webapp development and integration
- Async processing and error handling
- Monitoring metrics (success rate, latency, token usage)
- Cost tracking and allocation
- Audit logging and compliance
- Role-based access control (RBAC)

**Key Question Types:**
- API deployment (28%)
- Webapp integration (28%)
- Monitoring and governance (44%)

---

## Assessment Design Philosophy

All quizzes follow research-backed best practices:

### Question Design
- **Multiple-choice format** for objective assessment
- **Detailed explanations** for all options (correct and incorrect)
- **Scenario-based questions** testing real-world application
- **Dataiku-specific content** focusing on platform features
- **Progressive difficulty** from basic to advanced concepts

### Feedback Strategy
Each answer includes:
1. **Correct answer identification**
2. **Explanation of why it's correct**
3. **Explanation of why other options are incorrect**
4. **Additional context or best practices**

### Learning Objectives Alignment
Questions map directly to module learning objectives:
- **Knowledge** (recall facts and concepts)
- **Comprehension** (explain ideas and relationships)
- **Application** (use knowledge in scenarios)
- **Analysis** (troubleshoot and compare approaches)

### Performance Indicators
Each quiz provides:
- **Score ranges** with interpretation
- **Common mistakes** to avoid
- **Key concepts** to master
- **Next steps** for improvement

## Quiz Statistics

| Module | Questions | Base Points | Bonus Points | Time |
|--------|-----------|-------------|--------------|------|
| Module 0 | 12 | 100 | 0 | 20 min |
| Module 1 | 13 | 100 | 10 | 20 min |
| Module 2 | 14 | 100 | 18 | 20 min |
| Module 3 | 13 | 100 | 10 | 20 min |
| Module 4 | 14 | 100 | 0 | 20 min |
| **Total** | **66** | **500** | **38** | **100 min** |

## Topic Distribution

### By Learning Domain

| Domain | Questions | Percentage |
|--------|-----------|------------|
| Architecture & Setup | 16 | 24% |
| Prompt Engineering | 13 | 20% |
| RAG & Knowledge Banks | 14 | 21% |
| Python Development | 13 | 20% |
| Deployment & Ops | 14 | 21% |

### By Bloom's Taxonomy Level

| Level | Questions | Percentage |
|-------|-----------|------------|
| Remember | 15 | 23% |
| Understand | 25 | 38% |
| Apply | 18 | 27% |
| Analyze | 8 | 12% |

## Assessment Best Practices

### For Instructors

1. **Use as formative assessment**: Quizzes help identify learning gaps
2. **Allow multiple attempts**: Focus on learning, not gatekeeping
3. **Review common mistakes**: Use data to improve instruction
4. **Update with platform changes**: Keep content current with Dataiku updates

### For Students

1. **Complete after module study**: Use quizzes to validate understanding
2. **Review explanations**: Learn from both correct and incorrect answers
3. **Take notes on mistakes**: Create personal study guides
4. **Retake if needed**: Mastery is the goal, not first-attempt success

## Integration with Other Assessments

These quizzes complement other course assessments:

- **Hands-on notebooks**: Interactive coding exercises
- **Project assignments**: Multi-step application building
- **Peer reviews**: Code quality and documentation
- **Capstone project**: Comprehensive end-to-end application

## Future Enhancements

Potential improvements for subsequent course versions:

1. **Auto-grading integration**: Connect to Dataiku DSS for automated scoring
2. **Adaptive difficulty**: Adjust question difficulty based on performance
3. **Question banks**: Randomize questions for each attempt
4. **Time tracking**: Monitor time per question for difficulty calibration
5. **Performance analytics**: Detailed learner progress dashboards

## Accessibility Considerations

All quizzes are designed with accessibility in mind:
- Clear, concise language
- Sufficient time for reading and consideration
- Text-only format (no images required)
- Compatible with screen readers
- Multiple attempts allowed

## Academic Integrity

Quiz design includes measures to promote honest assessment:
- Explanations encourage learning over memorization
- Multiple attempts with feedback promote mastery
- Scenario-based questions test understanding, not recall
- Integration with hands-on work validates practical skills

---

## Contact & Feedback

For questions about assessments or to provide feedback:
- Review module README files for context
- Check CLAUDE.md for course design principles
- Refer to course_creator.md for assessment design framework

**Last Updated:** February 2, 2026
**Course Version:** 1.0
**Assessment Designer:** Claude (Assessment Designer Agent)
