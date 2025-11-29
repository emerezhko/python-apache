For Bitbucket Server (now Data Center), you have two main approaches. You should verify if the **built-in feature** meets your needs first, as it requires no coding. If you specifically need custom logic (e.g., checking specific statuses or custom fields) via a **Groovy script** (ScriptRunner), the second approach provides the code you requested.

### Approach 1: Native "Commit Checker" (No Code)
Bitbucket Data Center has a native feature that does exactly this.
1.  Go to **Repository Settings** > **Commit checker**.
2.  Enable **Require valid Jira issue keys**.
3.  Check the option **Check that the issue key exists in Jira**.
    *   *Note: This requires an Application Link between Bitbucket and Jira to be configured.*

---

### Approach 2: ScriptRunner Groovy Script (Custom Pre-Receive Hook)
If you need to implement this manually (e.g., for complex logic), you use a **Pre-Receive Hook**.

**Location:**
`Admin` -> `ScriptRunner` -> `Pre-Receive Hooks` -> `Custom Pre-Receive Hook`.

**The Logic:**
1.  Iterate over the `refChanges` (the branches/tags being pushed).
2.  Retrieve the commits associated with those changes.
3.  Extract the Jira Key using Regex (`[A-Z]+-\d+`).
4.  Use the `ApplicationLinkService` to call Jira's REST API and verify the issue exists.

**The Script:**

```groovy
import com.atlassian.bitbucket.commit.Commit
import com.atlassian.bitbucket.commit.CommitService
import com.atlassian.bitbucket.commit.CommitsBetweenRequest
import com.atlassian.bitbucket.hook.HookResponse
import com.atlassian.bitbucket.repository.RefChange
import com.atlassian.bitbucket.repository.RefChangeType
import com.atlassian.bitbucket.repository.Repository
import com.atlassian.sal.api.component.ComponentLocator
import com.atlassian.applinks.api.ApplicationLinkService
import com.atlassian.applinks.api.application.jira.JiraApplicationType
import com.atlassian.sal.api.net.Request
import com.atlassian.sal.api.net.Response
import com.atlassian.sal.api.net.ResponseException
import com.atlassian.sal.api.net.ResponseHandler
import com.atlassian.bitbucket.util.PageUtils
import java.util.regex.Matcher
import java.util.regex.Pattern

// Get necessary services
def commitService = ComponentLocator.getComponent(CommitService)
def appLinkService = ComponentLocator.getComponent(ApplicationLinkService)

// 1. Get the Primary Link to Jira
def jiraLink = appLinkService.getPrimaryApplicationLink(JiraApplicationType.class)
if (!jiraLink) {
    // Fail safe: if no Jira link exists, maybe allow or block depending on policy
    // hookResponse.out().println("Warning: No Jira Application Link found. Skipping validation.")
    return true 
}

// 2. Define Jira Key Regex (Standard: PROJECT-123)
Pattern jiraKeyPattern = Pattern.compile(/([A-Z][A-Z0-9]+-[0-9]+)/)

// 3. Helper function to check if issue exists in Jira
def issueExistsInJira(String issueKey, def applicationLink) {
    def exists = false
    try {
        // Call Jira REST API: /rest/api/2/issue/{issueKey}
        // We use createAuthenticatedRequestFactory to use the system auth
        def request = applicationLink.createAuthenticatedRequestFactory()
                .createRequest(Request.MethodType.GET, "/rest/api/2/issue/${issueKey}?fields=key")
        
        request.execute(new ResponseHandler<Response>() {
            @Override
            void handle(Response response) throws ResponseException {
                // 200 OK means found. 404 means not found.
                if (response.statusCode == 200) {
                    exists = true
                }
            }
        })
    } catch (Exception e) {
        // Log error if needed
    }
    return exists
}

// 4. Iterate over RefChanges (the push details)
for (RefChange refChange : refChanges) {
    // Skip deletions
    if (refChange.type == RefChangeType.DELETE) continue

    // Calculate new commits
    // If it's a new branch (ADD), we might want to check from 0000... to tip, 
    // but usually checking from default branch is safer. 
    // For simplicity, we assume standard flow or check the "toHash" commit.
    
    def commitsRequest = new CommitsBetweenRequest.Builder(repository)
            .exclude(refChange.fromHash) // Exclude old tip (commits already there)
            .include(refChange.toHash)   // Include new tip
            .build()

    // Retrieve commits
    def commits = commitService.getCommitsBetween(commitsRequest, PageUtils.newRequest(0, 999))
    
    for (Commit commit : commits.values) {
        def message = commit.message
        Matcher matcher = jiraKeyPattern.matcher(message)
        
        if (!matcher.find()) {
            // Block if no ID found
            resultBuilder.veto("Commit ${commit.displayId} rejected: No Jira Issue Key found in message.")
            return false
        }
        
        // Extract the Key
        String issueKey = matcher.group(1)
        
        // Verify existence in Jira
        if (!issueExistsInJira(issueKey, jiraLink)) {
            resultBuilder.veto("Commit ${commit.displayId} rejected: Jira Issue ${issueKey} does not exist or cannot be accessed.")
            return false
        }
    }
}

return true
```

### Key Components of the Script
1.  **`ApplicationLinkService`**: This is the most critical part. It allows the script to talk to your linked Jira instance securely without hardcoding passwords.
2.  **`CommitsBetweenRequest`**: This efficiently calculates which commits are *new* in this push (between the old hash and the new hash) so you don't re-scan old history.
3.  **`resultBuilder.veto()`**: This is the standard Bitbucket Server API method to block a push and display an error message to the user's Git client.

### Common Pitfalls
*   **Performance:** Making a REST call to Jira for *every* commit can be slow if a user pushes 50 commits at once. Consider caching results or bulk-checking if possible (though bulk API is more complex to script).
*   **New Branches:** When `refChange.type` is `ADD` (new branch), `refChange.fromHash` is a string of zeros (`000...`). Passing this to `exclude()` might throw an error or return too many commits. You may need to handle `ADD` by excluding the repository's default branch (e.g., `master` or `main`).
*   **Authentication:** The script uses `createAuthenticatedRequestFactory()`. This relies on the "Trusted Applications" or "OAuth" setup between your Bitbucket and Jira. If they are not linked with 2-legged auth (impersonation), this call might fail.
