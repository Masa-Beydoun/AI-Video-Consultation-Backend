# consulting/signals.py
from django.db.models.signals import post_migrate
from django.dispatch import receiver
from consulting.models import Domain, SubDomain



@receiver(post_migrate)
def create_default_domains(sender, **kwargs):
    if sender.name == "consulting":
        defaults = ["Tech", "Health", "Finance"]
        for name in defaults:
            Domain.objects.get_or_create(name=name)





@receiver(post_migrate)
def create_default_domains_and_subdomains(sender, **kwargs):
    if sender.name == 'consulting':  # Only run for this app

        # Create or get domains
        health_domain, _ = Domain.objects.get_or_create(name="Health")
        tech_domain, _ = Domain.objects.get_or_create(name="Tech")

        # Create default subdomains
        SubDomain.objects.get_or_create(name="General Health", domain=health_domain)
        SubDomain.objects.get_or_create(name="Fitness", domain=health_domain)
        SubDomain.objects.get_or_create(name="Software", domain=tech_domain)
        SubDomain.objects.get_or_create(name="Hardware", domain=tech_domain)
