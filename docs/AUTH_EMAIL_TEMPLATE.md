# URQEN signup confirmation email

The branded signup email is in [`supabase/templates/confirmation.html`](../supabase/templates/confirmation.html). [`supabase/config.toml`](../supabase/config.toml) references it for Supabase CLI/local projects. It explains that the message is from URQEN and includes Supabase's `{{ .ConfirmationURL }}` confirmation link.

**Important:** the files in this repository do not change the email sent by your already-hosted Supabase project by themselves. That is why you have not seen the new URQEN wording yet. To change the live signup email, open your Supabase project dashboard and apply it in **Authentication → Email Templates → Confirm signup**:

1. Set the subject to `Confirm your URQEN email address`.
2. Copy the HTML from `supabase/templates/confirmation.html` into the message body and save.

Hosted project email templates are managed in the Supabase dashboard (or its Management API); adding this repository template does not automatically change a hosted project's settings. For Supabase CLI/local projects, configure `auth.email.template.confirmation` in `supabase/config.toml` to reference this template file.
