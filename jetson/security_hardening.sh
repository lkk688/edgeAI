#!/bin/bash
# security_hardening.sh

echo "=== Jetson Security Hardening Script ==="
echo

# Function to check if running as root
check_root() {
    if [ "$EUID" -ne 0 ]; then
        echo "Please run as root (use sudo)"
        exit 1
    fi
}

# Function to backup configuration files
backup_config() {
    local file=$1
    if [ -f "$file" ]; then
        cp "$file" "${file}.backup.$(date +%Y%m%d_%H%M%S)"
        echo "Backed up $file"
    fi
}

# Update system
update_system() {
    echo "1. Updating system packages..."
    apt update && apt upgrade -y
    apt autoremove -y
    echo "System updated."
    echo
}

# Configure SSH security
secure_ssh() {
    echo "2. Securing SSH configuration..."
    backup_config "/etc/ssh/sshd_config"
    
    # SSH hardening
    sed -i 's/#PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
    sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config
    sed -i 's/#Protocol 2/Protocol 2/' /etc/ssh/sshd_config
    
    # Add additional security settings
    echo "" >> /etc/ssh/sshd_config
    echo "# Security hardening" >> /etc/ssh/sshd_config
    echo "MaxAuthTries 3" >> /etc/ssh/sshd_config
    echo "ClientAliveInterval 300" >> /etc/ssh/sshd_config
    echo "ClientAliveCountMax 2" >> /etc/ssh/sshd_config
    echo "X11Forwarding no" >> /etc/ssh/sshd_config
    
    systemctl restart ssh
    echo "SSH secured and restarted."
    echo
}

# Configure firewall
setup_firewall() {
    echo "3. Setting up UFW firewall..."
    
    # Install UFW if not present
    apt install -y ufw
    
    # Reset UFW to defaults
    ufw --force reset
    
    # Set default policies
    ufw default deny incoming
    ufw default allow outgoing
    
    # Allow SSH
    ufw allow 22/tcp
    
    # Allow common services (uncomment as needed)
    # ufw allow 80/tcp   # HTTP
    # ufw allow 443/tcp  # HTTPS
    # ufw allow 8080/tcp # Alternative HTTP
    
    # Enable firewall
    ufw --force enable
    
    echo "Firewall configured and enabled."
    echo
}

# Set up automatic security updates
setup_auto_updates() {
    echo "4. Setting up automatic security updates..."
    
    apt install -y unattended-upgrades
    
    # Configure unattended upgrades
    cat > /etc/apt/apt.conf.d/50unattended-upgrades << EOF
Unattended-Upgrade::Allowed-Origins {
    "\${distro_id}:\${distro_codename}-security";
    "\${distro_id}ESMApps:\${distro_codename}-apps-security";
    "\${distro_id}ESM:\${distro_codename}-infra-security";
};

Unattended-Upgrade::AutoFixInterruptedDpkg "true";
Unattended-Upgrade::MinimalSteps "true";
Unattended-Upgrade::Remove-Unused-Dependencies "true";
Unattended-Upgrade::Automatic-Reboot "false";
EOF

    # Enable automatic updates
    cat > /etc/apt/apt.conf.d/20auto-upgrades << EOF
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
EOF

    echo "Automatic security updates configured."
    echo
}

# Secure shared memory
secure_shared_memory() {
    echo "5. Securing shared memory..."
    
    backup_config "/etc/fstab"
    
    # Add tmpfs mount for /tmp if not present
    if ! grep -q "/tmp" /etc/fstab; then
        echo "tmpfs /tmp tmpfs defaults,noexec,nosuid,nodev,size=1G 0 0" >> /etc/fstab
    fi
    
    # Secure /dev/shm
    if ! grep -q "/dev/shm" /etc/fstab; then
        echo "tmpfs /dev/shm tmpfs defaults,noexec,nosuid,nodev 0 0" >> /etc/fstab
    fi
    
    echo "Shared memory secured."
    echo
}

# Set up fail2ban
setup_fail2ban() {
    echo "6. Setting up fail2ban..."
    
    apt install -y fail2ban
    
    # Configure fail2ban for SSH
    cat > /etc/fail2ban/jail.local << EOF
[DEFAULT]
bantime = 3600
findtime = 600
maxretry = 3

[sshd]
enabled = true
port = ssh
logpath = /var/log/auth.log
maxretry = 3
bantime = 3600
EOF

    systemctl enable fail2ban
    systemctl restart fail2ban
    
    echo "fail2ban configured and started."
    echo
}

# Main execution
main() {
    check_root
    
    echo "Starting security hardening process..."
    echo "This script will:"
    echo "1. Update system packages"
    echo "2. Secure SSH configuration"
    echo "3. Set up UFW firewall"
    echo "4. Configure automatic security updates"
    echo "5. Secure shared memory"
    echo "6. Set up fail2ban"
    echo
    
    read -p "Do you want to continue? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
    
    update_system
    secure_ssh
    setup_firewall
    setup_auto_updates
    secure_shared_memory
    setup_fail2ban
    
    echo "=== Security Hardening Complete ==="
    echo "Please review the changes and reboot the system."
    echo "Backup files have been created with timestamps."
    echo
    echo "Important notes:"
    echo "- SSH root login has been disabled"
    echo "- Password authentication has been disabled"
    echo "- Make sure you have SSH key access before rebooting"
    echo "- UFW firewall is now active"
    echo "- fail2ban is monitoring SSH attempts"
}

# Run main function
main "$@"