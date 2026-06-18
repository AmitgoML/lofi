"""Shared in-app navigation reference for Lucy agents.

Any agent that answers operational or navigational questions should include
LOFI_NAVIGATION_BLOCK in its system prompt so Lucy can link users directly to
the relevant page inside the Lofi app.

The frontend intercepts markdown links whose href starts with '/': they trigger
SPA navigation (keeping the Lucy panel open) instead of opening a new tab.
Standard markdown syntax is all that is needed — [label](/path).
"""

LOFI_NAVIGATION_BLOCK = """
## Lofi App Navigation
When your answer references a specific section of the Lofi application, include a markdown link using the internal path from the table below. Only use paths from this table — do not invent routes.

| Page                                  | Path                                    |
|---------------------------------------|-----------------------------------------|
| Dashboard                             | /                                       |
| Campaigns (list)                      | /campaigns                              |
| Create new campaign                   | /campaigns/new                          |
| Edit a specific campaign              | /campaigns/:id/edit                     |
| Relaunch a specific campaign          | /campaigns/:id/relaunch                 |
| Creative library                      | /creative                               |
| Help & documentation                  | /help                                   |
| Brand settings – Overview             | /settings/brand?section=overview        |
| Brand settings – Products             | /settings/brand?section=products        |
| Brand settings – Competitors          | /settings/brand?section=competitors     |
| Brand settings – Target Audiences     | /settings/brand?section=audiences       |
| Brand settings – Visuals              | /settings/brand?section=visuals         |
| Brand settings – Brand Messaging      | /settings/brand?section=messaging       |
| Brand settings – Blacklisted Keywords | /settings/brand?section=guidelines      |
| Brand settings – Locations            | /settings/brand?section=locations       |
| Ad account settings                   | /settings/ad-accounts                   |
| Integrations                          | /settings/integrations                  |
| Billing                               | /settings/billing                       |
| User management                       | /settings/user-management               |
| Account settings                      | /settings/account                       |

For dynamic campaign links, use the campaign ID from context or tool results:
- [View campaign](/campaigns/<campaign-id>)
- [Edit campaign](/campaigns/<campaign-id>/edit)

Only include a link when it is genuinely helpful and directly relevant to the user's question. Do not force links into every response.
"""
