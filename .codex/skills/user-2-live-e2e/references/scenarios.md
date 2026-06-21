# User 2.0 E2E Scenarios

## full

This is the default live regression suite. It mirrors the user's expected account flow in the Streamlit dashboard instead of behaving like an API debugging panel.

### Frontend Setup

- Open the Streamlit dashboard in the user's local Chrome session.
- Select `View` -> `Users`.
- Confirm the logged-out first screen presents `Login` and `Create account` entry points.
- Confirm each auth entry point can switch between `Email`, `Phone`, and `Simulated OAuth`.

### UI Steps

Use the values printed by:

```powershell
.\.venv\Scripts\python.exe .\.codex\skills\user-2-live-e2e\scripts\run_user_2_live_e2e.py --suite full --print-ui-scenario
```

Perform the printed steps in order:

- Create an email account.
- Update Profile fields.
- Change the email account password.
- Request and confirm password reset from the Security tab. The UI must keep the demo reset token internal and not show it.
- Link the provider that should remain connected.
- Link and then unlink the provider marked for unlink.
- Add both payment methods, make the second one default, and remove the first one.
- Subscribe to the printed plan and then cancel the subscription.
- Sign out, then log in again with the final reset password.
- Create and log in with the phone account.
- Sign out, then use Simulated OAuth login with the printed provider identity.

### Human-Visible Checks

- Logged-out users cannot see Profile, Security, Connected accounts, Billing, or Subscription account tools.
- Logged-in users see a short account summary and a `Sign out` action.
- The UI never renders `access_token`, `token_hash`, `reset_token`, `password_hash`, raw response JSON, or JSON validation bodies.
- Error messages are human-readable, for example "Email already registered" or "Invalid user credentials", not raw FastAPI JSON.
- Payment data is displayed as masked/tokenized metadata only.

### Machine Checks

The verifier checks final API state for the generated run id:

- Email login succeeds only with the final reset password.
- The old and intermediate email passwords fail.
- Profile fields match the expected update.
- The kept OAuth provider is connected and the unlinked provider is absent.
- The final active payment method is the expected default method.
- The removed payment method is not active.
- Subscription plan matches the printed plan and status is `cancelled`.
- Phone login succeeds for the generated phone account.
- User API responses do not contain password/session/reset token hashes.
- The saved report does not contain plaintext passwords or tokens.
