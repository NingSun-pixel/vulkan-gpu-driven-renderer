#include<rayTracing.h>



glm::vec3 Ray::at(float t) const
{
    return origin + dir * t;
}


void hit_record::set_face_normal(const Ray& r, const glm::vec3& outward_normal) {
    // Sets the hit record normal vector.
    // NOTE: the parameter `outward_normal` is assumed to have unit length.

    front_face = glm::dot(r.dir, outward_normal) < 0;
    normal = front_face ? outward_normal : -outward_normal;
}


bool sphere::hit(const Ray& r, double ray_tmin, double ray_tmax, hit_record& rec) const{
    glm::vec3 oc = center - r.origin;
    float a = glm::dot(r.dir, r.dir);
    float h = glm::dot(r.dir, oc);
    float c = glm::dot(oc, oc) - radius * radius;

    float discriminant = h * h - a * c;
    if (discriminant < 0)
        return false;

    float sqrtd = std::sqrt(discriminant);

    // Find the nearest root that lies in the acceptable range.
    float root = (h - sqrtd) / a;
    if (root <= ray_tmin || ray_tmax <= root) {
        root = (h + sqrtd) / a;
        if (root <= ray_tmin || ray_tmax <= root)
            return false;
    }

    rec.t = root;
    rec.p = r.at(rec.t);
    glm::vec3 outward_normal = (rec.p - center) / float(radius);
    rec.set_face_normal(r, outward_normal);

    return true;
}

bool hittable_list::hit(const Ray& r, double ray_tmin, double ray_tmax, hit_record& rec) const {
    hit_record temp_rec;
    bool hit_anything = false;
    auto closest_so_far = ray_tmax;

    for (const auto& object : objects) {
        if (object->hit(r, ray_tmin, closest_so_far, temp_rec)) {
            hit_anything = true;
            closest_so_far = temp_rec.t;
            rec = temp_rec;
        }
    }

    return hit_anything;
}