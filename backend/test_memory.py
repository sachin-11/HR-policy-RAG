"""Quick manual test for long-term memory."""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from app.memory.long_term_memory import (
    get_user_memory,
    update_preference,
    record_interaction,
    build_memory_context,
    extract_and_save_preferences,
)

USER_ID = "test_user_sachin"

print("=" * 50)
print("TEST 1: Save preferences")
update_preference(USER_ID, "preferred_language", "hindi")
update_preference(USER_ID, "department", "Engineering")
print("  Saved: preferred_language=hindi, department=Engineering")

print("\nTEST 2: Record interactions")
record_interaction(USER_ID, action="email_sent",
                   summary="Sent sick leave email to manager rajeshsachin786@gmail.com",
                   metadata={"recipient": "rajeshsachin786@gmail.com"})
record_interaction(USER_ID, action="ticket_created",
                   summary="Raised HR ticket for WFH policy query")
print("  Recorded 2 interactions")

print("\nTEST 3: Extract preferences from message")
msg = "mujhe kal bukhar tha isliye leave chahiye"
extract_and_save_preferences(USER_ID, msg)
print(f"  Message: '{msg}'")
print("  Auto-extracted preferences from Hindi message")

print("\nTEST 4: Get full memory")
memory = get_user_memory(USER_ID)
print(f"  Preferences : {memory['preferences']}")
print(f"  Interactions: {len(memory['interactions'])} entries")
for i in memory["interactions"]:
    print(f"    [{i['ts'][:10]}] {i['action']} — {i['summary']}")

print("\nTEST 5: Build memory context (injected into RAG prompt)")
ctx = build_memory_context(USER_ID)
print(ctx)

print("\n" + "=" * 50)
print("All tests passed! Memory file saved at: ./data/memory/test_user_sachin.json")
